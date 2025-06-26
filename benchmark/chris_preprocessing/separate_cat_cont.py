import pandas as pd
import os
import numpy as np

def encode_row(row):
    unique_values = set([val for val in row if val != -89])  # Exclude -89
    value_mapping = {val: i for i, val in enumerate(sorted(unique_values))}
    return [value_mapping[val] if val in value_mapping else val for val in row]

if __name__ == "__main__":
    in_path = "/home/fwoller/Projects/chris_data/chris_csv/"
    out_path = "/home/fwoller/Projects/chris_data/chris_csv/"
    missing_value = -89.0
    
    # Separate continuous and categorical data in phenotypes.
    pheno_data = pd.read_csv(in_path + 'phenotypes_data.csv')
    pheno_labels = pd.read_csv(in_path + 'phenotypes_labels.csv')
    
    # Open metabolites and protein data.
    metabolites = pd.read_csv(in_path + 'metabolites_data.csv')
    proteins = pd.read_csv(in_path + 'proteomics_data.csv')
    
    # Merge phenotypes, proteins, metabolites data.
    combined_data = pd.concat([pheno_data, metabolites, proteins], ignore_index=True)
    combined_data.drop(columns=[col for col in combined_data.columns if col.startswith('8')], inplace=True)

    # Translate NaN resulting from merge into missing value encoding.
    combined_data.fillna(missing_value, inplace=True)
    
    # Extract list of categorical variables.
    cat_labels = pheno_labels[pheno_labels['type'].isin(['categorical', 'boolean'])]
    cat_variables = set(cat_labels['label'])
    cont_labels = pheno_labels[pheno_labels['type'].isin(['float', 'integer'])]
    cont_variables = set(cont_labels['label'])
    
    # Separate phenotype data.
    cat_data = combined_data[combined_data['label'].isin(cat_variables)]
    cont_data = combined_data[~combined_data['label'].isin(cat_variables)]    
    
    # Map categorical values to start from 0 (ignoring missing value).
    cat_data_labels = cat_data['label'].tolist()
    cat_data.drop(columns=['label'], inplace=True)
    cat_data = cat_data.apply(encode_row, axis=1, result_type='expand')
    
    cont_data.drop(columns=['label'], inplace=True)
    cont_data.to_csv(out_path + 'napy_cont_data.csv')
    cat_data.to_csv(out_path + 'napy_cat_data.csv')
    
    # Select binary variables from categoricals.
    cat_temp = cat_data.replace(missing_value, np.nan)
    condition = cat_temp.apply(lambda row: row.dropna().isin([0, 1]).all(), axis=1)
    bin_data = cat_data[condition]
    bin_data.to_csv(out_path + 'napy_bin_data.csv')
    