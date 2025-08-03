# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_dir=Path('C:/git/BaekKhani_DCMSEAL/data').resolve()
keepraw_cols=['id', 'chid', 'alt', 'match','iv','ov','wk','wt','nwk','cost','tiv','ntiv','aux','tt','PS','nTrans']
potential_cats = ['purpose', 'hr', 'worktype', 'stu', 'engflu', 'age','income', 'disability', 'gender', 'choicerider']
test_size, random_state= 0.2, 5723588

def load_and_preprocess_data(data_dir: Path, keepraw_cols: list[str] = [], potential_cats: list[str] = [],
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
    cols_to_scale = []
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

    for col in numerical_cols:
        if col in keepraw_cols + one_hot_cols:
            continue
        if train_df[col].nunique() <= 2: #inherently binary
            continue
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

