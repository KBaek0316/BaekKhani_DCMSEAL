# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class ChoiceDataset(Dataset):
    """
    An optimized Dataset that separates per-alternative data from
    per-scenario (chid-level) data for maximum efficiency.
    """
    def __init__(self, dataframe: pd.DataFrame, x_emb_tensor: torch.Tensor, config: dict):
        super().__init__()
        
        # --- 1. Define variable types ---
        self.core_vars = config.get("core_vars", [])
        self.seg_vars = config.get("segmentation_vars", [])
        self.emb_vars = config.get("embedding_vars", [])

        # --- 2. Create the per-alternative DataFrame ---
        # This DataFrame only contains columns that change for each alternative.
        alt_cols = ['chid', 'alt', 'match'] + self.core_vars
        self.alt_df = dataframe[alt_cols].copy()

        # --- 3. Create the per-scenario (chid) inputs ---
        # These scenario_ objects contain only one row for each 'chid'.
        scenario_cols = ['chid'] + self.seg_vars
        self.chids, unique_indices = np.unique(dataframe['chid'], return_index=True)
        self.scenario_df = dataframe[scenario_cols].iloc[unique_indices,:].reset_index()
        self.scenario_x_emb = x_emb_tensor[unique_indices]

        # --- 4. Create the mapping to the alternative data: e.g., {78:[0,1,2,3,4],79:[5,6,7,8],...}---
        self.alt_indices_map = self.alt_df.groupby('chid').indices

    def __len__(self):
        return len(self.chids)

    def __getitem__(self, idx: int): #idx: an index from 0 to N-1
        target_chid = self.chids[idx] # index input to chid output

        # --- A. Get the single row of per-scenario data ---
        # Note: We use the main index `idx` here because self.scenario_df and self.chids are both sorted and aligned.
        seg_features = torch.tensor(self.scenario_df.loc[idx, self.seg_vars].values, dtype=torch.float32)
        x_emb = self.scenario_x_emb[idx]

        # --- B. Get the block of per-alternative data ---
        alt_row_indices = self.alt_indices_map[target_chid]
        start, end = alt_row_indices[0], alt_row_indices[-1] + 1 #e.g., (0, 5) for 78, (5, 9) for 79

        alt_data_slice = self.alt_df.iloc[start:end]
        core_features = torch.tensor(alt_data_slice[self.core_vars].values, dtype=torch.float32)
        choice = torch.tensor(alt_data_slice['match'].values.argmax(), dtype=torch.long)

        return {
            'core_features': core_features, # Shape: (J, C)
            'seg_features': seg_features,   # Shape: (S,)
            'x_emb': x_emb,                 # Shape: (E,)
            'choice': choice                # Sclaer: chosen alternative index from 0 to J-1)
        }

def choice_collate_fn(batch):
    #This pads per-alternative data while simply stacking the per-scenario data.

    # --- A. Pad the per-alternative data (core_features) ---
    core_features_list, masks_list = [], []
    max_len = max(item['core_features'].shape[0] for item in batch) # mostly J; "forward" developed to handle sub-J inputs

    for item in batch: #this loop runs B times
        core_feat = item['core_features'] # (J, C)
        num_alts = core_feat.shape[0] # num of rows: A
        padding_needed = max_len - num_alts # =J-A

        # `pad` takes (left,right,top,bottom); we only need to pad the bottom with ZEROs to make it J-rowed
        core_features_list.append(F.pad(core_feat, (0, 0, 0, padding_needed))) #(J, C)
        masks_list.append(torch.cat([torch.ones(num_alts, dtype=torch.bool), 
                                     torch.zeros(padding_needed, dtype=torch.bool)])) #appending (J,) vector
    #After the loop, core_features_list is length B list with (J,C) tensors and masks_lists len B with (J,)

    # --- B. Directly stack the per-scenario data ---
    # These are already single tensors, so we just collect and stack them.
    seg_features_list = [item['seg_features'] for item in batch] # len B list of (S,)
    x_emb_list = [item['x_emb'] for item in batch] # len B list of (E,)
    choices_list = [item['choice'] for item in batch] # len B list of scaler indices

    # --- C. Return the final, list length-becmae-the-first-dimension packed batch ---
    return {
        'core_features': torch.stack(core_features_list), # Shape: (B, J, C)
        'seg_features': torch.stack(seg_features_list), # Shape: (B, S)
        'x_emb': torch.stack(x_emb_list),                 # Shape: (B, E)
        'mask': torch.stack(masks_list),                  # Shape: (B, J) binary; 1-real 0-fake
        'choice': torch.stack(choices_list)               # Shape: (B,) choice alt indices (0 to J-1)
    }


#%% deprecated
class ChoiceDataset_flexible(Dataset): #second trial; deprecated
    """
    PyTorch Dataset for DCM-SEAL. This Dataset treats (converts) each choice scenario
    (a group of alternatives for one 'chid') as a single sample (i.e., wide format).
    dataframe and x_emb_tensor from data_processing.py; updated config from main.py
    """
    def __init__(self, dataframe: pd.DataFrame, x_emb_tensor: torch.Tensor, config: dict):
        super().__init__()
       
        self.df = dataframe
        self.x_emb = x_emb_tensor
        self.core_vars = config.get("core_vars", [])
        self.seg_vars = config.get("segmentation_vars", [])

        # We store the start and end index of each group in the original df.
        self.groups = self.df.groupby('chid').indices # e.g., {78: [0,1,2,3,4], 79: [0,1,2,3],...}
        self.chids = sorted(self.groups.keys())

    def __len__(self):
        """Returns the total number of choice scenarios."""
        return len(self.group_indices)

    def __getitem__(self, idx: int): #idx: an index from 0 to N-1
        """
        Retrieves one complete choice scenario using a robust mapping.
        """
        # --- Step 1: Map the DataLoader's index `idx` to the actual chid value ---
        target_chid = self.chids[idx]

        # --- Step 2: Use the chid to find the correct row indices ---
        row_indices = self.groups[target_chid]
        start_row, end_row = row_indices[0], row_indices[-1] + 1 #e.g., (0,5) for 78, (5,8) for 79

        # --- Step 3: Slice the data using the correct row indices ---
        scenario_df = self.df.iloc[start_row:end_row]
        scenario_x_emb = self.x_emb[start_row:end_row]

        # --- Step 4: Convert to tensors assume this chid has A<=J alternatives or rows---
        core_features = torch.tensor(scenario_df[self.core_vars].values, dtype=torch.float32) #(A, C)
        seg_features = torch.tensor(scenario_df[self.seg_vars].values, dtype=torch.float32) #(A, S)
        choice = torch.tensor(scenario_df['match'].values.argmax(), dtype=torch.long) # scaler: chosen alt index
        
        return {
            'core_features': core_features,
            'seg_features': seg_features,
            'x_emb': scenario_x_emb,
            'choice': choice
        }


def choice_collate_fn_flexible(batch): #second trial associated with above class; deprecated
    """
    Custom collate function to handle choice scenarios of variable lengths.

    This function takes a list of samples (df-like dictionaries), each from ChoiceDataset class
    and pads them to ensure uniformity (J) with masking info, finally to get a B-length batch.

    Args:
        batch (list[dict]): A list of dictionaries, where each dictionary
                            represents one choice scenario from __getitem__.

    Returns:
        dict: A dictionary containing batched and padded tensors for features,
              choices, and a mask to identify the padded elements.
    """
    # Find the maximum number of alternatives in this batch for padding
    max_len = max(item['core_features'].shape[0] for item in batch) # mostly J; "forward" developed to handle sub-J inputs

    # Prepare lists to hold padded tensors for each key
    core_features_list, seg_features_list, x_emb_list, masks_list, choices_list = [], [], [], [], []

    # Iterate through each sample (a choice scenario) in the batch to grow lists
    for item in batch: # Runs B times
        # --- 1. Get the data for the current scenario ---
        core_feat = item['core_features'] #(A, C)
        seg_feat = item['seg_features'] #(A, S)
        x_emb = item['x_emb'] #(A, E)
        
        # --- 2. Calculate how much padding is needed ---
        num_alternatives = core_feat.shape[0] # num of rows: A
        padding_needed = max_len - num_alternatives # =J-A

        # --- 3. Pad each feature tensor to max_len ---
        # `pad` takes (padding_left, padding_right, padding_top, padding_bottom)
        # We only need to pad the "bottom" (the alternatives dimension) with zeros to make it J-rowed
        core_features_list.append(torch.nn.functional.pad(core_feat, (0, 0, 0, padding_needed))) # (J,C)
        seg_features_list.append(torch.nn.functional.pad(seg_feat, (0, 0, 0, padding_needed))) # (J,S)
        x_emb_list.append(torch.nn.functional.pad(x_emb, (0, 0, 0, padding_needed), value=0)) # (J,E)

        # --- 4. Create a boolean mask ---
        # `True` for real data, `False` for padding. The model uses this to ignore padded values.
        mask = torch.cat([torch.ones(num_alternatives, dtype=torch.bool), 
                          torch.zeros(padding_needed, dtype=torch.bool)]) #(J, )
        masks_list.append(mask)
        
        # --- 5. Collect the choice index ---
        choices_list.append(item['choice']) # appending scaler choice index

    # --- 6. Stack the lists of tensors into a single batch tensor ---
    # The final dictionary that the model will receive in the training loop
    return {
        'core_features': torch.stack(core_features_list), # (B, J, C) data
        'seg_features': torch.stack(seg_features_list), # (B, J, S) data
        'x_emb': torch.stack(x_emb_list), # (B, J, E) data
        'mask': torch.stack(masks_list), # (B, J) binary
        'choice': torch.stack(choices_list) # (B,) indices
    }


class ChoiceDatasetByRow(Dataset): #first trial, deprecated
    """
    Custom PyTorch Dataset for the DCM-SEAL model.

    This class takes the pre-processed pandas DataFrame and the embedding tensor
    and prepares them for consumption by the PyTorch DataLoader. It defines how
    to retrieve a single data point and format it into the dictionary structure
    that the model's `forward` method expects.
    """
    def __init__(self, dataframe: pd.DataFrame, x_emb_tensor: torch.Tensor, config: dict):
        """
        Initializes the Dataset.

        Args:
            dataframe (pd.DataFrame): The pre-processed dataframe containing all
                                      variables except the embedding features.
            x_emb_tensor (torch.Tensor): The (B, E) integer tensor for the
                                         embedding variables.
            config (dict): The configuration dictionary, used to get the lists
                           of core and segmentation variable names.
        """
        self.df = dataframe
        self.x_emb = x_emb_tensor

        # Get all variable names that the model will need from the dataframe
        self.core_vars = config.get("core_vars", [])
        seg_vars_cat = config.get("segmentation_vars_categorical", [])
        seg_vars_cont = config.get("segmentation_vars_continuous", [])
        
        # After one-hot encoding, the categorical variable names have changed.
        # We need to find the new one-hot encoded column names.
        self.seg_vars = seg_vars_cont + [
            col for col in dataframe.columns if any(f"{s}_" in col for s in seg_vars_cat)
        ]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves one sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary where keys are feature names and values are
                  the corresponding data as PyTorch tensors. This is the 'batch'
                  that the model's forward() method will receive.
        """
        # Get the row from the dataframe
        row = self.df.iloc[idx]
        
        # Start building the output dictionary
        data_dict = {
            # --- Key identifiers ---
            'chid': torch.tensor(row['chid'], dtype=torch.long),
            'alt': torch.tensor(row['alt'], dtype=torch.long),
            'match': torch.tensor(row['match'], dtype=torch.long),
            
            # --- Embedding variables tensor ---
            'x_emb': self.x_emb[idx, :]
        }
        
        # --- Add core and segmentation variables ---
        for var_name in self.core_vars:
            data_dict[var_name] = torch.tensor(row[var_name], dtype=torch.float32)
            
        for var_name in self.seg_vars:
            data_dict[var_name] = torch.tensor(row[var_name], dtype=torch.float32)
            
        return data_dict

if __name__ == '__main__':
    pass