import pandas as pd
import numpy as np
import math
import os

# mapping -> CHRIS internal IDs to descriptions/proteinIDs/...
# labels -> data with type of variables (categorical, min, max, missing,...)
# data -> actual data


path = "/data/chrisportal/datasets/3_valid/"

# Exclude time-dependent, irrelevant modules, and modules that contain several data entries for one sample.
exclude = ["cc_longitudinal_info", "x0_scanningswath_peptides", "x0_scanningswath_proteins", "x0_drugs", 
           "x0_pedigree", "x0_genomics", "x1_general", "cc_biochemical_traits", "cc_vaccination", 
           "x0_metabolomics_mxp500", "cc_baseline_info", "cc_general_info", "cc_neutralizing_antibody"]

proteins = ["x0_somalogic"]

metabolites = ["x0_metabolomics_p180"]

out_path = "/home/fwoller/Projects/chris_data/chris_csv/"

ID = "Sample_ID"

missing_value = -89.0

def process_proteins_tdff(path, dir):
    # check if the directory contains a folder with
    version = "1.0.3.2"
    variable_with_aid = "aid"

    dir_path = path + dir + "/" + version + "/data/"

    if os.path.exists(dir_path):
        # read data file
        data = pd.read_csv(dir_path + "data.txt", sep="\t")

        # read groups file to only get the protein abundance data
        groups = pd.read_csv(dir_path + "groups.txt", sep="\t")
        # extract rows of groups where group column == protein_abundance
        groups = groups[groups["group"] == "protein_abundance"]

        # subset data to only the variables in label of groups and the variable with the aid
        data = data.loc[:, [variable_with_aid] + groups["label"].tolist()]

        # labels_additional_info.txt -> for Protein ID, EntrezGeneID, EntrezGeneSymbol
        mapping = pd.read_csv(dir_path + "labels_additional_info.txt", sep="\t")
        cols = ["label", "long_description", "TargetFullName", "Target", "UniProt", "EntrezGeneID", "EntrezGeneSymbol"]
        mapping = mapping.loc[:,cols]
        # only keep the rows where the label is in the data
        mapping = mapping[mapping["label"].isin(data.columns)]

        # labels -> subset to data
        label = pd.read_csv(dir_path + "labels.txt", sep="\t")
        label = label[label["label"].isin(data.columns)]

        # rename the variable with the aid to ID
        data.set_index(variable_with_aid, inplace=True)
        data = data.T
        data = data.reset_index()
        data.rename(columns={'index': 'label'}, inplace=True)
        # just remove row with variable with aid from mapping and loc
        mapping = mapping[mapping["label"] != variable_with_aid]
        label = label[label["label"] != variable_with_aid]
        
        # Map missing value to chosen missing value encoder.
        proteins_missing = -89.0
        data.replace(proteins_missing, missing_value, inplace=True)

        return data, mapping, label

    else:
        print("No directory:" + dir, "with version: " + version)
        
def process_metabolites_tdff(path, dir):
    # check if the directory contains a folder with
    version = "1.0.1.2"

    dir_path = path + dir + "/" + version + "/data/"

    if os.path.exists(dir_path):
        # read data file
        data = pd.read_csv(dir_path + "data.txt", sep="\t")

        # read groups file to only get the metabolite assay concentration data
        groups = pd.read_csv(dir_path + "groups.txt", sep="\t")
        groups = groups[groups["group"] == "assay_concentrations"]

        # subset data to only the variables in label of groups and the variable with the aid
        data = data.loc[:, ["aid"] + groups["label"].tolist()]

        # labels_additional_info.txt -> for metabolite name and IDs
        mapping = pd.read_csv(dir_path + "labels_additional_info.txt", sep="\t")
        cols = ["label", "long_description", "analyte_name", "analyte_class", "analyte_quant", "biochemical_name", "hmdb_id"]
        mapping = mapping.loc[:,cols]
        # only keep the rows where the label is in the data
        mapping = mapping[mapping["label"].isin(data.columns)]

        # labels -> subset to data
        label = pd.read_csv(dir_path + "labels.txt", sep="\t")
        label = label[label["label"].isin(data.columns)]

        # rename the variable with the aid to ID
        data.set_index("aid", inplace=True)
        data = data.T
        data = data.reset_index()
        data.rename(columns={'index': 'label'}, inplace=True)
        # just remove row with variable with aid from mapping and loc
        mapping = mapping[mapping["label"] != "aid"]
        label = label[label["label"] != "aid"]
        
        # Map missing value to chosen missing value encoder.
        metabolites_missing = -89.0
        data.replace(metabolites_missing, missing_value, inplace=True)

        return data, mapping, label

    else:
        print("No directory:" + dir, "with version: " + version)
        
        

def process_tdff(path, dir):
    # check if the directory contains a folder with
    version = "1.0.0.2" # Apparently all "phenotypes" have version 1.0.0.2

    dir_path = path + dir + "/" + version + "/data/"

    if os.path.exists(dir_path):
        # Read data file
        data = pd.read_csv(dir_path + "data.txt", sep="\t")
        
        # Filter out variables with character as data type.
        labels = pd.read_csv(dir_path + "labels.txt", sep='\t')
        labels = labels[~labels["type"].isin(["character", "date", "time"])]
        data = data.loc[:, ["aid"] + labels["label"].tolist()]
        
        # Read mapping file and only keep those that are also present in subsetted data.
        mapping = pd.read_csv(dir_path + "mapping.txt", sep='\t')
        mapping = mapping[mapping["label"].isin(data.columns)]
        
        # Rename the variable with the aid to ID
        data = data.rename(columns={"aid": "label"})
        column_counts = data["label"].value_counts()
        duplicates = column_counts[column_counts > 1]
        
        # Number of duplicate column names
        num_duplicates = duplicates.sum()
        # print("Number of duplicates in sample IDs: ", num_duplicates)
        if num_duplicates > 0:
            print("Detected duplicates in " + dir)
            print(duplicates)
            
        # Set sample IDs as index.
        data.set_index("label", inplace=True)
        data = data.T
        
        # Print format (i.e. lenght) of Sample IDs.
        id_length = len(str(data.columns[0]))
        print("ID lenght in directory " + dir +" is: ", id_length)
        
        # Map all missing values to one common missing value encoding.
        # Extract list of float & integer variables.
        cont_labels = labels[labels['type'].isin(['float', 'integer'])]
        cont_data = data.loc[list(set(cont_labels['label']))]
        # Iterate over all float & integer variables.
        for index, row in cont_data.iterrows():
            # Look up min max values of variable.
            variable = str(index)
            var_min = cont_labels.loc[cont_labels['label'] == variable, 'min'].iloc[0]
            var_max = cont_labels.loc[cont_labels['label'] == variable, 'max'].iloc[0]
            # Map all values below min and above max to missing value.
            cont_data.loc[index, (cont_data.loc[index] < var_min) | (cont_data.loc[index] > var_max)] = missing_value
        
        # Extract list of categorical & boolean variables.
        cat_labels = labels[labels['type'].isin(['categorical', 'boolean'])]
        cat_data = data.loc[list(set(cat_labels['label']))]
        # Iterate over all categorical variables.
        for index, row in cat_data.iterrows():
            # Check for specific x0lp61d variable that contains meaningful negative categories.
            variable = str(index)
            if variable == "x0lp61d":
                # Only treat value of -89 as missing.
                cat_data.loc[index, (cat_data.loc[index]==-89.0)] = missing_value
            else:
                # Map all negative values to missing.
                cat_data.loc[index, (cat_data.loc[index] < 0)] = missing_value
        
        # Concatenate transformed cat and cont dataframes.
        data = pd.concat([cont_data, cat_data], ignore_index=False)
        data = data.reset_index()
        data.rename(columns={'index': 'label'}, inplace=True)  
        
        return data, mapping, labels       

    else:
        print("No directory:" + dir, "with version: " + version)


pheno_data = pd.DataFrame()
pheno_labels = pd.DataFrame()
pheno_mapping = pd.DataFrame()
for dir in os.listdir(path):
    # check if directory is in exclude list -> don't process
    if dir in exclude:
        continue

    if dir in proteins:
        data, mapping, label = process_proteins_tdff(path, dir)
        data.to_csv(out_path + "proteomics_data.csv", index=False)
        mapping.to_csv(out_path + "proteomics_mapping.csv", index=False)
        label.to_csv(out_path + "proteomics_label.csv", index=False)
        continue

    if dir in metabolites:
        data, mapping, label = process_metabolites_tdff(path, dir)
        data.to_csv(out_path + "metabolites_data.csv", index=False)
        mapping.to_csv(out_path + "metabolites_mapping.csv", index=False)
        label.to_csv(out_path + "metabolites_label.csv", index=False)
        continue
    
    # Otherwise module belongs to phenotypes.
    data, mapping, labels = process_tdff(path, dir)
    
    # Check if phenotype dataframes need to be initialized at first iteration.
    if pheno_data.empty:
        pheno_data = data.copy()
        pheno_labels = labels.copy()
        pheno_mapping = mapping.copy()
        pheno_data.to_csv(out_path + "pheno_small.csv", index=False)
    # Append data, labels and mapping to current status.
    else:
        pheno_labels = pd.concat([pheno_labels, labels], ignore_index=True)
        pheno_mapping = pd.concat([pheno_mapping, mapping], ignore_index=True)
        pheno_data = pd.concat([pheno_data, data], ignore_index=True)   
        
# Save phenotype data to files and replace NA resulting form concatenation.
pheno_data.index.name = None
pheno_data.fillna(missing_value, inplace=True)
pheno_data.to_csv(out_path + "phenotypes_data.csv", index=False)
pheno_labels.to_csv(out_path + "phenotypes_labels.csv", index=False)
pheno_mapping.to_csv(out_path + "phenotypes_mapping.csv", index=False)
