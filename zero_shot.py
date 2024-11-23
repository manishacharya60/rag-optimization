import json
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set up logging to track errors
logging.basicConfig(filename='zero_error_log.log', level=logging.ERROR)

# Array to store IDs of entries that encountered errors
error_ids = []

# Function to call GPT-4o for code optimization (zero-shot)
def call_gpt4_optimization(code_to_be_optimized):
    prompt = f"""
    You are an expert code optimization assistant. Given the code below, optimize it as efficiently as possible. Only return the optimized code itself, with no additional information, comments, or explanations.

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
        
        # Remove ```cpp markers from the beginning and end, if present
        if optimized_code.startswith("```cpp"):
            optimized_code = optimized_code[5:].strip()
        if optimized_code.endswith("```"):
            optimized_code = optimized_code[:-3].strip()
        
        # Extra cleaning step: remove any unwanted characters at the start, such as stray letters
        optimized_code = optimized_code.lstrip("p\n ")

        return optimized_code
    except Exception as e:
        logging.error(f"Error processing optimization. Exception: {e}")
        print(f"Error processing optimization. Exception: {e}")
        return None

# Main function to iterate through test_dataset and find the optimized code
def main(test_dataset_file, output_file):
    # Load test dataset
    with open(test_dataset_file, 'r') as file:
        test_dataset = {entry['id']: entry for entry in json.load(file)}

    # Store optimization results
    optimization_results = []

    # Run for all entries in test_dataset
    for index, (test_id, entry) in enumerate(test_dataset.items()):
        code_to_be_optimized = entry['source_code']
        groundtruth_code = entry['optimized_code']  # Retrieve ground truth optimized code

        # Generate optimized code using GPT-4o API
        generated_code = call_gpt4_optimization(code_to_be_optimized)

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
        print(f"Processed {index + 1}/{len(test_dataset)}")

    # Save optimization results with ground truth
    with open(output_file, 'w') as f:
        json.dump(optimization_results, f, indent=4)

    # Report error IDs
    if error_ids:
        print("\nThe following IDs encountered errors:")
        print(error_ids)
    else:
        print("\nAll entries processed successfully!")

# Run the main function
if __name__ == "__main__":
    # Define file paths
    test_dataset_file = "path/to/your/dataset"
    output_file = "path/to/your/output"

    # Execute main function
    main(test_dataset_file, output_file)
