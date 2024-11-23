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
logging.basicConfig(filename='context_error_log.log', level=logging.ERROR)

# Array to store IDs of entries that encountered errors
error_ids = []

# Function to call GPT-4 for code optimization
def call_gpt4_optimization(code_to_be_optimized, similar_source_code, similar_optimized_code, cfg_labels, code_analysis_text):
    prompt = f"""
    You are an expert in C++ code optimization, with a deep understanding of optimizing code by analyzing Control Flow Graphs (CFGs) and implementing efficient coding practices. Your task is to generate an optimized version of the provided code using the reference code and its optimized counterpart as examples, along with insights from their CFG differences and analysis.

**Context of CFG and CFG Differences:**
The CFG represents the sequence and conditions in which code blocks are executed, showing all possible paths that execution could take in the code. CFG differences indicate how the control flow has changed between the original and optimized versions, highlighting areas where execution can be streamlined or redundancies reduced. Use these differences to identify redundant operations, unnecessary branches, or bottlenecks in the provided code.

**Instructions for Using CFG Differences and Analysis to Optimize Code:**
1. Look for repetitive code segments or loops in the CFG that have been condensed or removed in the optimized reference. Apply similar techniques to minimize redundant calculations or loops in the code to optimize.
2. Focus on eliminating unnecessary branches and simplifying conditions where CFG shows reduced nodes in the optimized version.
3. Where possible, reduce the number of intermediate variables, limit memory usage, and improve inlining to match the optimized CFG structure.

Use the CFG differences, labels, and analysis provided to make structural and algorithmic improvements in the code that will yield a faster, more memory-efficient version. 

**Reference Code:**
Original Code:
{similar_source_code}

Optimized Code:
{similar_optimized_code}

CFG Differences/Labels:
{cfg_labels}

Analysis:
{code_analysis_text}

**Code to Optimize:**
{code_to_be_optimized}

Return only the optimized code, with no additional comments or explanations.
    """

    try:
        # Call GPT-4 API
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

# Main function to iterate through test_cfg_embeddings and find the optimized code
def main(test_cfg_embeddings_file, train_cfg_embeddings_file, test_dataset_file, train_dataset_file, train_cfg_dataset_file, code_analysis_file, output_file):
    # Load embeddings and datasets
    test_cfg_embeddings = pd.read_csv(test_cfg_embeddings_file)
    train_cfg_embeddings = pd.read_csv(train_cfg_embeddings_file)

    with open(test_dataset_file, 'r') as file:
        test_dataset = {entry['id']: entry for entry in json.load(file)}

    with open(train_dataset_file, 'r') as file:
        train_dataset = {entry['id']: entry for entry in json.load(file)}

    with open(train_cfg_dataset_file, 'r') as file:
        train_cfg_dataset = {entry['id']: entry for entry in json.load(file)}

    with open(code_analysis_file, 'r') as file:
        code_analysis = {entry['id']: entry['analysis'] for entry in json.load(file)}

    # Convert embeddings to numpy arrays using source_cfg_embeddings
    test_cfg_embeddings['source_cfg_embeddings'] = test_cfg_embeddings['source_cfg_embeddings'].apply(lambda x: np.array(json.loads(x)))
    train_cfg_embeddings['source_cfg_embeddings'] = train_cfg_embeddings['source_cfg_embeddings'].apply(lambda x: np.array(json.loads(x)))

    # Store optimization results
    optimization_results = []
    total_entries = len(test_cfg_embeddings)

    # Iterate over each entry in test_cfg_embeddings
    for index, test_entry in test_cfg_embeddings.iterrows():
        test_id = test_entry['id']
        code_to_be_optimized = test_dataset[str(test_id)]['source_code']
        groundtruth_code = test_dataset[str(test_id)]['optimized_code']  # Retrieve ground truth optimized code

        # Find the most similar training example using source_cfg_embeddings
        test_embedding = test_entry['source_cfg_embeddings'].reshape(1, -1)
        train_embeddings = np.stack(train_cfg_embeddings['source_cfg_embeddings'].values)
        similarities = cosine_similarity(test_embedding, train_embeddings)
        most_similar_index = similarities.argmax()
        most_similar_id = train_cfg_embeddings.iloc[most_similar_index]['id']

        # Retrieve data from training examples
        similar_source_code = train_dataset[str(most_similar_id)]['source_code']
        similar_optimized_code = train_dataset[str(most_similar_id)]['optimized_code']
        cfg_labels = train_cfg_dataset[str(most_similar_id)]['labels']
        code_analysis_text = code_analysis[str(most_similar_id)]

        # Generate optimized code using GPT-4 API
        generated_code = call_gpt4_optimization(
            code_to_be_optimized,
            similar_source_code,
            similar_optimized_code,
            cfg_labels,
            code_analysis_text
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
    test_cfg_embeddings_file = "path/to/your/test/embeddings"
    train_cfg_embeddings_file = "path/to/your/train/embeddings"
    test_dataset_file = "path/to/your/test/dataset"
    train_dataset_file = "path/to/your/train/embeddings"
    train_cfg_dataset_file = "path/to/your/train/cfg"
    code_analysis_file = "path/to/your/analysis"
    output_file = "path/to/your/output"

    # Execute main function
    main(test_cfg_embeddings_file, train_cfg_embeddings_file, test_dataset_file, train_dataset_file, train_cfg_dataset_file, code_analysis_file, output_file)
