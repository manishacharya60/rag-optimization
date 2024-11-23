import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set up logging to keep track of errors
logging.basicConfig(filename='error_log.log', level=logging.ERROR)

# Array to store IDs of entries that encountered errors
error_ids = []

def call_gpt4_analysis(source_code, optimized_code, labels):
    prompt = f"""
    Act as a software optimization expert. Based on the provided source code and optimized code, as well as the described changes between their control flow graphs (CFGs), analyze the key transformations made during the optimization process. Focus on how these changes (provided in the form of labels) highlight structural and functional improvements. Provide insights into the rationale behind the optimizations, how they reduce complexity or improve performance, and how similar transformations can be applied to optimize other code.
    
    Source Code: {source_code}
    Optimized Code: {optimized_code}
    Changes (Labels): {labels}
    """

    try:
        # Call GPT-4o using the updated API format
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can generate the best analysis for given codes."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract GPT-4o's response text correctly
        analysis = completion.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        # Log the error
        logging.error(f"Error processing entry. Exception: {e}")
        print(f"Error processing entry with ID: {id}. Exception: {e}")
        return None  # Return None if there's an error

def main(train_dataset_file, train_cfg_dataset_file, output_file):
    # Load train_dataset.json
    with open(train_dataset_file, 'r') as f:
        train_dataset = json.load(f)

    # Load train_cfg_dataset.json
    with open(train_cfg_dataset_file, 'r') as f:
        train_cfg_dataset = json.load(f)

    # Create list to store analysis results
    analysis_results = []
    total_entries = len(train_dataset)
    processed_entries = 0

    # Iterate over entries in the train_dataset
    for train_entry in train_dataset:
        entry_id = train_entry.get("id")
        
        # Find matching entry in train_cfg_dataset based on "id"
        matching_entry = next((cfg_entry for cfg_entry in train_cfg_dataset if cfg_entry.get("id") == entry_id), None)
        
        if matching_entry:
            # Extract necessary fields
            source_code = train_entry.get("source_code")
            optimized_code = train_entry.get("optimized_code")
            labels = matching_entry.get("labels")  # Use labels instead of CFGs

            # Generate analysis using GPT-4 API
            analysis = call_gpt4_analysis(source_code, optimized_code, labels)

            # Check if the analysis was generated successfully
            if analysis is not None:
                # Store result with "id" and "analysis"
                analysis_results.append({
                    "id": entry_id,
                    "analysis": analysis
                })
            else:
                # Store the failed entry ID and print the error
                error_ids.append(entry_id)
                logging.error(f"Skipped entry with id {entry_id} due to an error.")
                print(f"Error for ID {entry_id}")

        # Increment the processed entry count and print the progress
        processed_entries += 1
        print(f"Processed {processed_entries}/{total_entries}")

    # Write the analysis results to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=4)

    # Print the array of error IDs at the end
    if error_ids:
        print("\nThe following IDs encountered errors:")
        print(error_ids)
    else:
        print("\nAll entries were processed successfully!")

if __name__ == "__main__":
    # Specify input and output file paths
    train_dataset_file = "path/to/your/train/dataset"
    train_cfg_dataset_file = "path/to/your/train/cfg"
    output_file = "path/to/your/output/analysis"

    # Call main function
    main(train_dataset_file, train_cfg_dataset_file, output_file)
