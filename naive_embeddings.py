import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set up logging to track errors
logging.basicConfig(filename='naive_error_log.log', level=logging.ERROR)

# Array to store IDs of entries that encountered errors
error_ids = []

# Function to call GPT-4o for code optimization
def call_gpt4_optimization(code_to_be_optimized, similar_source_code, similar_optimized_code):
    prompt = f"""
    You are an expert C++ code optimization assistant. Given the reference code and optimization example, optimize the following code. Only return the optimized code itself, with no additional information, comments, or explanations.

    Reference Code:
    Original Code:
    {similar_source_code}
    Optimized Code:
    {similar_optimized_code}

    Code to Optimize:
    {code_to_be_optimized}
    """

    try:
        # Call GPT-4o API
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a highly skilled assistant providing optimized code without additional explanations."},
                {"role": "user", "content": prompt}
            ],
        )
        optimized_code = completion.choices[0].message.content.strip()
        return optimized_code
    except Exception as e:
        logging.error(f"Error processing optimization. Exception: {e}")
        print(f"Error processing optimization. Exception: {e}")
        return None

# Main function to iterate through source code embeddings and find the optimized code
def main(test_source_code_embeddings_file, train_source_code_embeddings_file, test_dataset_file, train_dataset_file, output_file):
    # Load embeddings and datasets
    test_source_code_embeddings = pd.read_csv(test_source_code_embeddings_file)
    train_source_code_embeddings = pd.read_csv(train_source_code_embeddings_file)

    with open(test_dataset_file, 'r') as file:
        test_dataset = {str(entry['id']): entry for entry in json.load(file)}

    with open(train_dataset_file, 'r') as file:
        train_dataset = {str(entry['id']): entry for entry in json.load(file)}

    # Convert embeddings to numpy arrays
    test_source_code_embeddings['source_code_embeddings'] = test_source_code_embeddings['source_code_embeddings'].apply(lambda x: np.array(json.loads(x)))
    train_source_code_embeddings['source_code_embeddings'] = train_source_code_embeddings['source_code_embeddings'].apply(lambda x: np.array(json.loads(x)))

    # Store optimization results
    optimization_results = []
    total_entries = len(test_source_code_embeddings)

    # Iterate over each entry in test_source_code_embeddings
    for index, test_entry in test_source_code_embeddings.iterrows():
        test_id = test_entry['id']
        code_to_be_optimized = test_dataset[str(test_id)]['source_code']
        groundtruth_code = test_dataset[str(test_id)]['optimized_code']  # Retrieve ground truth optimized code

        # Find the most similar training example using source_code embeddings
        test_embedding = test_entry['source_code_embeddings'].reshape(1, -1)
        train_embeddings = np.stack(train_source_code_embeddings['source_code_embeddings'].values)
        similarities = cosine_similarity(test_embedding, train_embeddings)
        most_similar_index = similarities.argmax()
        most_similar_id = train_source_code_embeddings.iloc[most_similar_index]['id']

        # Retrieve data from training examples
        similar_source_code = train_dataset[str(most_similar_id)]['source_code']
        similar_optimized_code = train_dataset[str(most_similar_id)]['optimized_code']

        # Generate optimized code using GPT-4o API
        generated_code = call_gpt4_optimization(
            code_to_be_optimized,
            similar_source_code,
            similar_optimized_code
        )

        # Check if optimization was successful
        if generated_code is not None:
            optimization_results.append({
                "id": test_id,
                "code_to_be_optimized": code_to_be_optimized,
                "generated_code": generated_code,
                "groundtruth_code": groundtruth_code
            })
        else:
            error_ids.append(test_id)
            logging.error(f"Error processing entry with ID {test_id}")

        # Print progress
        print(f"Processed {index + 1}/{total_entries}")

    # Save optimization results with ground truth
    with open(output_file, 'w') as f:
        json.dump(optimization_results, f, indent=4)

    # Report error IDs
    if error_ids:
        print("\nThe following IDs encountered errors:")
        print(error_ids)
    else:
        print("\nAll entries were processed successfully!")

# Run the main function
if __name__ == "__main__":
    # Define file paths
    test_source_code_embeddings_file = "path/to/your/test/embeddings"
    train_source_code_embeddings_file = "path/to/your/train/embeddings"
    test_dataset_file = "path/to/your/test/dataset"
    train_dataset_file = "path/to/your/train/dataset"
    output_file = "path/to/your/output"

    # Execute main function
    main(test_source_code_embeddings_file, train_source_code_embeddings_file, test_dataset_file, train_dataset_file, output_file)
