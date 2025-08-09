# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import OrderedDict

'''
data_dir=Path('C:/git/BaekKhani_DCMSEAL/data').resolve()
keepraw_cols=['id', 'alt', 'match','iv','ov','wk','wt','nwk','cost','tiv','ntiv','aux','tt','PS','nTrans']
potential_cats = ['purpose', 'hr', 'worktype', 'stu', 'engflu', 'age','income', 'disability', 'gender', 'choicerider']
test_size, random_state= 0.2, 5723588
'''
def load_and_preprocess_data(config: dict, data_dir: Path, fname: str = ''):
    """
    Loads, preprocesses, and splits the data for the DCM-SEAL model.
    It first examines dfConv.csv to preprocess data based on domain knowledge manually given
    Then, it 
    It takes a list of embedding variables, discovers their cardinality (number of unique categories)
    after applying any consolidation rules, and returns this information for model initialization.

    Args:
        config (dict): A dictionary defining the data structure. Expected keys:
            - 'core_vars' (list[str]): Columns for core utility variables.
            - 'embedding_vars' (list[str]): A list of column names to be
              treated as embedding variables.
            - 'segmentation_vars_categorical' (list[str]): Categorical columns
              for the segmentation network (will be one-hot encoded).
            - 'segmentation_vars_continuous' (list[str]): Continuous columns
              for the segmentation network (will be standardized).
            - 'test_size' (float): Proportion for the test set.
            - 'random_state' (int): Seed for the train/test split.
        data_dir (Path): The path to the main 'data' directory.

    Returns:
        tuple: A tuple containing:
            - train_df (pd.DataFrame): The processed training dataframe.
            - test_df (pd.DataFrame): The processed testing dataframe.
            - train_x_emb (torch.Tensor): The integer tensor for embedding vars (training).
            - test_x_emb (torch.Tensor): The integer tensor for embedding vars (testing).
            - embedding_dims (OrderedDict): The discovered dimensions of the embedding variables.
    """
    # --- Unpack config ---
    core_vars = config.get("core_vars", [])
    emb_vars = config.get("embedding_vars", [])
    seg_vars_cat = config.get("segmentation_vars_categorical", [])
    seg_vars_cont = config.get("segmentation_vars_continuous", [])
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 5723588)

    # --- 1. Load data ---
    print(f"Loading data from {fname}")
    folder_name = data_dir / fname
    path_file = folder_name / "dfIn.csv"
    conv_file = folder_name / "dfConv.csv"
    try:
        df_in = pd.read_csv(path_file)
    except FileNotFoundError:
        print(f"Error: Could not find the main data file at {path_file}.")
        return None, None, None, None, None

    # --- 2. Apply contextual conversion rules, if a conversion file exists ---
    if conv_file.is_file():
        print("Conversion file found. Applying rules...")
        df_conv = pd.read_csv(conv_file)
        for field in df_conv['field'].unique():
            if field in df_in.columns:
                mapping = df_conv[df_conv['field'] == field].set_index('orilevel')['newlevel'].to_dict()
                df_in[field] = df_in[field].replace(mapping)
    else:
        print("No dfConv.csv found. Skipping categorical level conversion.")

    # --- 3. Discover Embedding Dims & Integer (label) Encode ---
    embedding_dims = OrderedDict() #need this to get trained weight matrix later
    for var in emb_vars:
        # Treat the column as categorical first
        df_in[var] = df_in[var].astype(str)
        # Now, apply the LabelEncoder to the string-based categories
        le = LabelEncoder()
        df_in[var] = le.fit_transform(df_in[var])
        # Discover the number of unique categories for this variable
        num_categories = len(le.classes_)
        embedding_dims[var] = num_categories
    print(f"Discovered embedding dimensions and integer encoding: {embedding_dims}")

    # --- 4. One-Hot Encode ONLY Segmentation Variables ---
    cols_to_encode = [col for col in seg_vars_cat if col in df_in.columns]
    df_processed = pd.get_dummies(df_in, columns=cols_to_encode, dummy_na=False)

    # --- 5. Split Data by Group (by 'id') ---
    print("Splitting training/test data by 'id'; Need to do this before standardization to prevent data leakage")
    unique_ids = df_processed['id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_df = df_processed[df_processed['id'].isin(train_ids)].copy()
    test_df = df_processed[df_processed['id'].isin(test_ids)].copy()

    # --- 6. Create Embedding Tensors ---
    if emb_vars: # if exists
        train_x_emb = torch.tensor(train_df[emb_vars].values, dtype=torch.long)
        test_x_emb = torch.tensor(test_df[emb_vars].values, dtype=torch.long)
    else:
        train_x_emb = torch.empty((len(train_df), 0), dtype=torch.long)
        test_x_emb = torch.empty((len(test_df), 0), dtype=torch.long)

    # --- 7. Standardize Numerical Columns ---
    print(f"Standardizing numerical columns, excluding cores: {core_vars}")
    scaler = StandardScaler()
    cols_to_scale = [v for v in seg_vars_cont if v in train_df.columns]

    if cols_to_scale:
        train_df[cols_to_scale] = train_df[cols_to_scale].astype('float64')
        test_df[cols_to_scale] = test_df[cols_to_scale].astype('float64')
        scaler.fit(train_df[cols_to_scale])
        train_df.loc[:, cols_to_scale] = scaler.transform(train_df[cols_to_scale])
        test_df.loc[:, cols_to_scale] = scaler.transform(test_df[cols_to_scale])
        print(f"Standardized columns: {cols_to_scale}")

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print("Preprocessing complete.")
    return train_df, test_df, train_x_emb, test_x_emb, embedding_dims


def load_and_preprocess_data_old(data_dir: Path, keepraw_cols: list[str] = [], potential_cats: list[str] = [],
                             test_size: float = 0.2, random_state: int = 5723588):
    """
    Loads, preprocesses, standardizes, and splits the raw survey data.

    Args:
        data_dir (Path): The path to the main 'data' directory.
        core_cols_to_exclude (list[str], optional): A list of columns that should
            NOT be standardized (e.g., core utility variables). Defaults to None.
        test_size (float): The proportion of the dataset for the test set.
        random_state (int): A seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Processed training and testing DataFrames.
        
    For Debugging:
        data_dir=Path('C:/git/BaekKhani_DCMSEAL/data').resolve()
        keepraw_cols=['id', 'alt', 'match','iv','ov','wk','wt','nwk','cost','tiv','ntiv','aux','tt','PS','nTrans']
        potential_cats = ['purpose', 'hr', 'worktype', 'stu', 'engflu', 'age','income', 'disability', 'gender', 'choicerider']
        test_size, random_state= 0.2, 5723588
    """
    # Define file paths
    empirical_dir = data_dir / "empirical"
    path_file = empirical_dir / "dfPath.csv"
    conv_file = empirical_dir / "dfConv.csv"

    # --- 1. Load Data ---
    print("Loading data...")
    try:
        df_path = pd.read_csv(path_file)
        df_conv = pd.read_csv(conv_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your data files are in 'data/empirical/'.")
        return None, None

    # --- 2. Apply Conversion Rules ---
    # Iterate through the conversion rules defined in dfConv.csv
    print("Applying conversion rules...")
    for field in df_conv['field'].unique():
        if field in df_path.columns:
            # Create a mapping dictionary from the conversion rules
            mapping = df_conv[df_conv['field'] == field].set_index('orilevel')['newlevel'].to_dict()
            df_path[field] = df_path[field].replace(mapping)

    # --- 3. One-Hot Encoding ---
    # Identify columns that are categorical (object type or low cardinality)
    # Note: 'id' and 'alt' are identifiers, not features for encoding.
    # 'match' is the choice outcome.
    # Note: 'id', 'alt', 'match' are identifiers, in addition to the columns defined in keepraw_cols.
    cols_before_dummy = set(df_path.columns)
    candidates = set(df_path.select_dtypes(include=['object', 'category']).columns)
    if len(potential_cats)>0:
        candidates.update(set([col for col in potential_cats if col in df_path.columns]))
    cols_to_encode = list(candidates-set(keepraw_cols))
    df_processed = pd.get_dummies(df_path, columns=cols_to_encode, dummy_na=False)
    print(f"One-hot encoded columns: {cols_to_encode}")
    # Get column names AFTER and find the new ones
    cols_after_dummy = set(df_processed.columns)
    one_hot_cols = list(cols_after_dummy - cols_before_dummy)

    # --- Step 4. Split Data by Group ---
    unique_ids = df_processed['id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_df = df_processed[df_processed['id'].isin(train_ids)].copy()
    test_df = df_processed[df_processed['id'].isin(test_ids)].copy()

    # --- Step 5. Standardize Numerical Columns (now using the robust one_hot_cols list) ---
    # --- Step 5. Standardize Numerical Columns ---
    cols_to_scale = []
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

    for col in numerical_cols:
        if col in keepraw_cols + one_hot_cols:
            continue
        if train_df[col].nunique() <= 2: #inherently binary
            continue
        if train_df[col].nunique() <= 2: 
            continue # pass inherently binary integer columns
        cols_to_scale.append(col)

    print(f"Columns identified for scaling: {cols_to_scale}")

    if len(cols_to_scale)>0:
        int_cols = train_df[cols_to_scale].select_dtypes(include='int64').columns #if you don't do this you'll see dtype warnings
        for col in int_cols:
            train_df[col] = train_df[col].astype('float64')
            test_df[col] = test_df[col].astype('float64')
        scaler = StandardScaler()
        scaler.fit(train_df[cols_to_scale]) #You always fit on the training data only
        train_df.loc[:, cols_to_scale] = scaler.transform(train_df[cols_to_scale])
        test_df.loc[:, cols_to_scale] = scaler.transform(test_df[cols_to_scale])
        print("Standardization complete.")
    else:
        print("No columns required scaling.")

    return train_df, test_df

if __name__ == '__main__':
    # This block allows you to test the script directly
    # Assumes the script is in src/ and the data is in data/
    project_dir = Path(__file__).resolve().parents[1]
    data_directory = project_dir / "data"
    train_data, test_data = load_and_preprocess_data(data_directory)
    if train_data is not None and test_data is not None:
        print("\n--- Training Data Head ---")
        print(train_data.head())
        print(f"\nTraining data shape: {train_data.shape}")

        print("\n--- Testing Data Head ---")
        print(test_data.head())
        print(f"\nTesting data shape: {test_data.shape}")
