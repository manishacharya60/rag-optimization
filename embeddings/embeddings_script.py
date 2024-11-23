import json
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel

# Load the JSON data from two files
with open('path/to/your/dataset', 'r') as file1, open('path/to/your/cfg', 'r') as file2:
    source_code_data = json.load(file1)
    source_cfg_data = json.load(file2)

# Initialize the tokenizer and model for CodeBERT
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Prepare the data for the CSV file
rows = []

# Create a dictionary for fast lookup of cfg entries by id
cfg_data_dict = {entry['id']: entry['source_cfg'] for entry in source_cfg_data}

# Iterate over the entries and compute embeddings if IDs match
for source_entry in source_code_data:
    source_id = source_entry['id']
    
    # Check if the id exists in both source_code and source_cfg data
    if source_id in cfg_data_dict:
        source_code = source_entry['source_code']
        source_cfg = cfg_data_dict[source_id]
        
        # Get embeddings for both source_code and source_cfg
        source_code_embedding = get_embedding(source_code)
        source_cfg_embedding = get_embedding(source_cfg)
        
        # Store the embeddings in a row format
        rows.append({
            'id': source_id,
            'source_code_embeddings': source_code_embedding,
            'source_cfg_embeddings': source_cfg_embedding
        })

# Convert to DataFrame for CSV saving
df = pd.DataFrame(rows)

# Save the DataFrame to CSV
df.to_csv('path/to/your/output', index=False)

print("Embeddings saved!")
