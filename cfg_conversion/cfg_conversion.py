import os
import subprocess
import tempfile
import json
import re

class BasicBlock:
    def __init__(self, name):
        self.name = name
        self.statements = []
        self.successors = []
        self.predecessors = []

    def __repr__(self):
        return (f"BasicBlock({self.name}, "
                f"statements={self.statements}, "
                f"successors={self.successors}, "
                f"predecessors={self.predecessors})")

def preprocess_code(code_str):
    # Replace #include <bits/stdc++.h> with specific headers
    code_str = re.sub(
        r'#include\s*<bits/stdc\+\+\.h>',
        '''
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <deque>
#include <list>
#include <functional>
#include <numeric>
#include <utility>
#include <limits>
#include <iomanip>  // For setprecision
using namespace std;
''',
        code_str
    )

    # Remove 'register' keyword
    code_str = re.sub(r'\bregister\b', '', code_str)

    # Ensure 'setprecision' is available
    if 'setprecision' in code_str and '<iomanip>' not in code_str:
        code_str = code_str.replace('#include <iostream>', '#include <iostream>\n#include <iomanip>')

    # Remove GCC attributes
    code_str = re.sub(r'__attribute__\s*\(\(.*?\)\)', '', code_str)

    return code_str

def get_cfg_output(code_str):
    """
    Use Clang's static analyzer to dump CFGs.
    """
    # Save code_str to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cpp', mode='w') as temp_file:
        temp_file.write(code_str)
        temp_filename = temp_file.name

    # Run Clang command to get CFG output
    clang_cmd = [
        'clang++',
        '-std=c++14',
        '-w',  # Suppress all warnings
        '-Xclang', '-analyze',
        '-Xclang', '-analyzer-checker=debug.DumpCFG',
        '-fsyntax-only',  # Ensure only syntax checking is done
        temp_filename
    ]
    try:
        result = subprocess.run(clang_cmd, capture_output=True, text=True)
        cfg_output = result.stdout + result.stderr  # Combine both outputs
        if result.returncode != 0:
            print(f"Clang returned non-zero exit code {result.returncode}")
            print(f"Clang error output:\n{cfg_output}")
            cfg_output = ''  # Discard CFG output if Clang failed
    except Exception as e:
        print(f"Error running Clang: {e}")
        cfg_output = ''
    finally:
        # Clean up temporary file
        os.unlink(temp_filename)

    return cfg_output

def parse_cfg_output(cfg_output):
    blocks = {}
    current_block = None

    lines = cfg_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('***'):
            continue  # Skip separator lines
        elif line.startswith('CFG for'):
            continue  # Skip CFG header lines
        elif re.match(r'^\[B\d+( \(.*\))?\]', line):
            # New basic block
            block_name = line.split()[0].strip('[]')
            current_block = BasicBlock(block_name)
            blocks[block_name] = current_block
        elif current_block is not None:
            if line.startswith('Preds') or line.startswith('Succs'):
                key, rest = line.split(':', 1)
                block_refs = re.findall(r'\[B\d+( \(.*\))?\]', rest)
                block_names = [ref.strip('[]').split()[0] for ref in block_refs]
                if key.startswith('Preds'):
                    current_block.predecessors = block_names
                elif key.startswith('Succs'):
                    current_block.successors = block_names
            else:
                # Statement
                current_block.statements.append(line)
        else:
            continue  # Ignore lines outside blocks

    return blocks

def compare_cfgs(blocks1, blocks2):
    labels = []
    differences_found = False

    # Get sorted lists of block names for consistent comparison
    all_block_names = set(blocks1.keys()).union(set(blocks2.keys()))

    for block_name in sorted(all_block_names):
        block1 = blocks1.get(block_name)
        block2 = blocks2.get(block_name)

        if block1 and not block2:
            labels.append(f"Block {block_name} removed in optimized code")
            differences_found = True
            continue
        if block2 and not block1:
            labels.append(f"Block {block_name} added in optimized code")
            differences_found = True
            continue

        # Compare statements
        statements1 = block1.statements
        statements2 = block2.statements

        if statements1 != statements2:
            differences_found = True
            labels.append(f"Block {block_name} statements changed")
            for idx in range(max(len(statements1), len(statements2))):
                stmt1 = statements1[idx] if idx < len(statements1) else "<no statement>"
                stmt2 = statements2[idx] if idx < len(statements2) else "<no statement>"
                if stmt1 != stmt2:
                    labels.append(f"Block {block_name}, statement {idx+1} changed from '{stmt1}' to '{stmt2}'")
            if len(statements1) != len(statements2):
                labels.append(f"Block {block_name} statement count changed from {len(statements1)} to {len(statements2)}")

        # Compare successors
        if block1.successors != block2.successors:
            differences_found = True
            labels.append(f"Block {block_name} successors changed from {block1.successors} to {block2.successors}")

        # Compare predecessors
        if block1.predecessors != block2.predecessors:
            differences_found = True
            labels.append(f"Block {block_name} predecessors changed from {block1.predecessors} to {block2.predecessors}")

    if not differences_found:
        labels.append("No differences detected between source and optimized CFGs")

    return labels

def main():
    # Load the dataset
    with open('../path/to/dataset', 'r') as f:
        data = json.load(f)

    new_data = []

    for idx, entry in enumerate(data):
        id = entry['id']
        source_code = preprocess_code(entry['source_code'])
        optimized_code = preprocess_code(entry['optimized_code'])

        print(f"Processing entry {idx+1}/{len(data)}: ID {id}")

        # Get CFGs
        source_cfg_output = get_cfg_output(source_code)
        optimized_cfg_output = get_cfg_output(optimized_code)

        # Check if CFG outputs are empty
        if not source_cfg_output.strip():
            print(f"Error: No CFG output for source code of ID {id}")
            continue  # Skip this entry

        if not optimized_cfg_output.strip():
            print(f"Error: No CFG output for optimized code of ID {id}")
            continue  # Skip this entry

        # Parse CFGs
        source_blocks = parse_cfg_output(source_cfg_output)
        optimized_blocks = parse_cfg_output(optimized_cfg_output)

        # Compare CFGs
        labels = compare_cfgs(source_blocks, optimized_blocks)

        # Include CFG outputs in the entry
        new_entry = {
            'id': id,
            'labels': labels,
            'source_cfg': source_cfg_output,
            'optimized_cfg': optimized_cfg_output
        }

        new_data.append(new_entry)

    # Save the new dataset
    with open('path/to/your/output', 'w') as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    main()
