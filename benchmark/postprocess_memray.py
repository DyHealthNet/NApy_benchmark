import os
import pandas as pd
import subprocess
import re

def split_filename(filename):
    # Remove the file extension (.txt) and split by underscore
    parts = filename.replace('.txt', '').split('_')
    return parts

def extract_memory_allocated(file_path):
    """Extracts the total memory allocated from a memray stats results file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'Total memory allocated:' in line:
            memory_allocated = lines[i + 1].strip()
            return memory_allocated
    return None  # Return None if the pattern is not found

def process_directory(directory_path):
    """Processes all files in a directory and stores memory allocation results in a pandas DataFrame."""
    results = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Only process txt files, ignore directories.
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            memory_allocated = extract_memory_allocated(file_path)
            if memory_allocated:  # Add to results if memory value is found
                results.append({'filename': filename, 'total_memory_allocated': memory_allocated})

    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results)
    return df

def process_memray_files(directory_path):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.bin'):
            # Construct full path to the .bin file
            bin_file_path = os.path.join(directory_path, filename)

            # Create the corresponding .txt file name by replacing .bin with .txt
            txt_filename = filename.replace('.bin', '.txt')
            txt_file_path = os.path.join(directory_path, txt_filename)

            # Call the 'memray stats' command and redirect output to the .txt file
            with open(txt_file_path, 'w') as output_file:
                subprocess.run(['memray', 'stats', bin_file_path], stdout=output_file)
                print(f'Processed {bin_file_path} -> {txt_file_path}.')

def process_df(df):
    out_df = pd.DataFrame(columns=['library', 'method', 'features', 'samples', 'na_ratio', 'threads', 'memory'])
    for index, row in df.iterrows():
        filename = row['filename']
        memory = row['total_memory_allocated']
        # Chop filename into parameter parts.
        parts = split_filename(filename)[:-1]
        parts.append(memory)
        new_row = pd.DataFrame([parts], columns=out_df.columns)
        out_df = pd.concat([out_df, new_row], ignore_index=True)

    return out_df

if __name__ == '__main__':
    # Example usage
    SKIP_MEMRAY = False
    directory_path = './'  # Replace with your directory path

    if not SKIP_MEMRAY:
        process_memray_files(directory_path)

    df = process_directory(directory_path)

    # Replace file names column with input parameters.
    df = process_df(df)
    df.to_csv('memory_results.csv', index=False)
