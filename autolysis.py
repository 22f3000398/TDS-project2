import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json


from google.colab import userdata

os.environ["AIPROXY_TOKEN"] = userdata.get('OPENAI_API_KEY')

# Function to summarize the dataset
def summarize_dataset(dataframe):
    print("Summarizing the dataset...")  # Debugging line
    # Summary statistics for numerical columns
    summary = dataframe.describe()

    # Check for missing values
    missing_data = dataframe.isnull().sum()

    # Select only numeric columns for correlation matrix
    numeric_data = dataframe.select_dtypes(include=[np.number])

    # Correlation matrix for numerical columns
    correlations = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()

    print("Summary complete.")  # Debugging line
    return summary, missing_data, correlations

# Function to identify outliers using IQR
def identify_outliers(dataframe):
    print("Identifying outliers...")  # Debugging line
    # Select only numeric columns
    numeric_data = dataframe.select_dtypes(include=[np.number])

    # Apply the IQR method to find outliers in the numeric columns
    lower_quartile = numeric_data.quantile(0.25)
    upper_quartile = numeric_data.quantile(0.75)
    interquartile_range = upper_quartile - lower_quartile
    detected_outliers = ((numeric_data < (lower_quartile - 1.5 * interquartile_range)) | 
                         (numeric_data > (upper_quartile + 1.5 * interquartile_range))).sum()

    print("Outlier identification complete.")  # Debugging line
    return detected_outliers

# Function to create visualizations (correlation heatmap, outlier plot, and distribution plot)
def create_visuals(correlations, detected_outliers, dataframe, output_folder):
    print("Creating visualizations...")  # Debugging line
    # Generate a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    heatmap_path = os.path.join(output_folder, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    # Check if there are outliers to plot
    if not detected_outliers.empty and detected_outliers.sum() > 0:
        # Plot the outliers
        plt.figure(figsize=(10, 6))
        detected_outliers.plot(kind='bar', color='red')
        plt.title('Outlier Detection')
        plt.xlabel('Columns')
        plt.ylabel('Count of Outliers')
        outlier_path = os.path.join(output_folder, 'outliers_plot.png')
        plt.savefig(outlier_path)
        plt.close()
    else:
        print("No outliers to visualize.")
        outlier_path = None  # No file created for outliers

    # Generate a distribution plot for the first numeric column
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_column = numeric_columns[0]  # Get the first numeric column
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[first_column], kde=True, color='blue', bins=30)
        plt.title(f'{first_column} Distribution')
        dist_plot_path = os.path.join(output_folder, f'{first_column}_distribution.png')
        plt.savefig(dist_plot_path)
        plt.close()
    else:
        dist_plot_path = None  # No numeric columns to plot

    print("Visualizations created.")  # Debugging line
    return heatmap_path, outlier_path, dist_plot_path

# Function to generate a story using an AI API
def generate_story_via_ai(prompt, analysis_context, max_words=1000):
    print("Generating story using AI...")  # Debugging line
    try:
        # Retrieve the proxy token from the environment variable
        proxy_token = os.environ["AIPROXY_TOKEN"]

        # Define the API URL for the proxy
        proxy_api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        # Compose the complete prompt
        detailed_prompt = f"""
        Based on the following analysis, craft a creative story. Ensure it has an engaging structure with an introduction, main content, and conclusion.

        Analysis Context:
        {analysis_context}

        Prompt:
        {prompt}

        The story should include:
        - A structured narrative with transitions.
        - Highlight the significance of the data.
        - Conclude with insights or reflections.
        """

        # Configure headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {proxy_token}"
        }

        # Create the request payload
        payload = {
            "model": "gpt-4o-mini",  # Specify the model for the proxy
            "messages": [
                {"role": "system", "content": "You are an insightful assistant."},
                {"role": "user", "content": detailed_prompt}
            ],
            "max_tokens": max_words,
            "temperature": 0.7
        }

        # Send the request
        response = requests.post(proxy_api_url, headers=headers, data=json.dumps(payload))

        # Verify the response
        if response.status_code == 200:
            # Extract and return the generated story
            story_content = response.json()['choices'][0]['message']['content'].strip()
            print("Story successfully generated.")  # Debugging line
            return story_content
        else:
            print(f"Request error: {response.status_code} - {response.text}")
            return "Error generating story."

    except Exception as error:
        print(f"An error occurred: {error}")
        return "Error generating story."

# Main workflow function
def workflow_main(data_file):
    print("Initiating the workflow...")  # Debugging line

    # Attempt to read the CSV file with compatible encoding
    try:
        dataframe = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Data loaded successfully.")  # Debugging line
    except UnicodeDecodeError as load_error:
        print(f"Failed to load data: {load_error}")
        return

    summary, missing_data, correlations = summarize_dataset(dataframe)
    detected_outliers = identify_outliers(dataframe)

    output_folder = "."
    os.makedirs(output_folder, exist_ok=True)

    # Generate visualizations
    heatmap_path, outlier_path, dist_plot_path = create_visuals(correlations, detected_outliers, dataframe, output_folder)

    # Prepare context summary for AI
    context_details = f"""
    Analysis Overview:
    Key Statistics (First 5 Columns):
    {summary.iloc[:, :5]}

    Missing Values:
    {missing_data.head()}

    Correlations (First 5 Columns):
    {correlations.iloc[:, :5]}

    Outliers Summary (First 5 Columns):
    {detected_outliers.head()}
    """

    # Use AI to generate a descriptive story
    story_result = generate_story_via_ai("Craft an interesting story from the data insights.", 
                                        analysis_context=context_details, 
                                        max_words=1000)

    print(f"Generated Story:\n{story_result}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_dataset>")
        sys.exit(1)
    workflow_main(sys.argv[1])
