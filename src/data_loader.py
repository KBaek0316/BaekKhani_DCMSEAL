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

class PaddedChoiceDataset(Dataset):
    """
    Precomputes per-chid tensors once so __getitem__ is O(1) and the default collate can just stack. Shapes:
      - core_bank:  (N, J_max, C)
      - mask_bank:  (N, J_max)              bool
      - choice_bank:(N,)                    long
      - seg_bank:   (N, S)                  float32
      - x_emb_bank: (N, E)                  long
    Legend: N=chids, J_max=max #alts per chid, C=#core vars, S=#seg vars, E=#embedding vars
    """
    def __init__(self, dataframe: pd.DataFrame, x_emb_tensor: torch.Tensor, config: dict):
        super().__init__()

        # Define core, segmentation, and embedding variables
        self.core_vars = config.get("core_vars", [])
        self.seg_vars = config.get("segmentation_vars", [])
        self.emb_vars = config.get("embedding_vars", [])
        self.J_max = config["n_alternatives"]  # (max J)

        if len(self.core_vars) == 0:
            raise ValueError("config['core_vars'] is empty; the model expects at least one core variable.")
        if 'chid' not in dataframe.columns or 'alt' not in dataframe.columns or 'match' not in dataframe.columns:
            raise ValueError("DataFrame must contain columns: 'chid', 'alt', 'match'.")

        # --- group once, keep deterministic chid order by first appearance ---
        group = dataframe.groupby('chid', sort=False)              # pandas groups keep order of first appearance
        keys = list(group.indices.keys())                          # chids in encounter order
        first_rows = [group.indices[k][0] for k in keys]           # first row index per chid
        order = np.argsort(first_rows)                             # ensure stable ascending by first index
        self.chids = np.array([keys[i] for i in order])            # (N,)
        first_idx  = np.array([first_rows[i] for i in order])      # (N,)

        # --- banks with one row per chid ---
        if self.seg_vars:
            self.seg_bank = torch.tensor(dataframe.iloc[first_idx][self.seg_vars].values, dtype=torch.float32) # (N, S)
        else:
            self.seg_bank = torch.empty((len(self.chids), 0), dtype=torch.float32)  # (N, 0)
        self.x_emb_bank = x_emb_tensor[first_idx]                  # (N, E)


        # --- preallocate per-alt banks ---
        N = len(self.chids)
        C = len(self.core_vars)
        core_bank  = torch.zeros((N, self.J_max, C), dtype=torch.float32)  # (N, J, C)
        mask_bank  = torch.zeros((N, self.J_max),    dtype=torch.bool)     # (N, J)
        choice_bank= torch.zeros((N,),               dtype=torch.long)     # (N,)

        # Populate pre-padded tensors using vectorized group indexing
        index_map = group.indices  # dict: chid -> np.ndarray of row positions
        for i, chid in enumerate(self.chids):
            idxs = index_map[chid]  # consecutive row indices for this chid
            block = dataframe.iloc[idxs]
            J_i = len(block)
            if J_i > self.J_max:
                raise ValueError(f"Found chid={chid} with {J_i} alternatives > J_max={self.J_max}. ")
            core_block = torch.tensor(block[self.core_vars].values, dtype=torch.float32)  # (J_i, C)
            core_bank[i, :J_i, :] = core_block # inject above to the i-th elem of (N,J,C) tensor
            mask_bank[i, :J_i] = True # inject J-1 valid alt=1 to (N, J)
            choice_bank[i] = int(block['match'].values.argmax())  # inject chosen alt index in [0..J_i-1] to (N,)

        self.core_bank = core_bank
        self.mask_bank = mask_bank
        self.choice_bank = choice_bank

    def __len__(self):
        return len(self.chids)

    def __getitem__(self, idx: int):
        # Tensors are precomputed during init. this method only slices
        return {
            'core_features': self.core_bank[idx],  # (J, C)
            'seg_features': self.seg_bank[idx],    # (S,)
            'x_emb': self.x_emb_bank[idx],         # (E,)
            'mask': self.mask_bank[idx],           # (J,)
            'choice': self.choice_bank[idx],       # ()
        }

#%% deprecated
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


class ChoiceDatasetByRow(Dataset): #"long" data assumed; deprecated
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