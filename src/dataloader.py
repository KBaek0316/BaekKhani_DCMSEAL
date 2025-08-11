# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


class ChoiceDataset(Dataset):
    """
    PyTorch Dataset for DCM-SEAL.
    This Dataset treats each choice scenario (a group of alternatives for one 'chid')
    as a single sample, which is the correct approach for this modeling task.
    """
    def __init__(self, dataframe: pd.DataFrame, x_emb_tensor: torch.Tensor, config: dict):
        super().__init__()
       
        self.df = dataframe
        self.x_emb = x_emb_tensor
        self.core_vars = config.get("core_vars", [])
        self.seg_vars = config.get("segmentation_vars", [])

        # --- Group data by 'chid' to create scenarios ---
        # We store the start and end index of each group in the original df.
        self.group_indices = []
        # find_groups returns a dictionary mapping chid to its indices
        groups = self.df.groupby('chid').indices
        for chid in sorted(groups.keys()): # Iterate in a sorted order
            start, stop = groups[chid][0], groups[chid][-1] + 1
            self.group_indices.append((start, stop))

    def __len__(self):
        """Returns the total number of choice scenarios."""
        return len(self.group_indices)

    def __getitem__(self, idx):
        """Retrieves one complete choice scenario."""
        # Get the start and end index for this scenario
        start, stop = self.group_indices[idx]
        
        # Slice the DataFrame and the embedding tensor for this specific scenario
        scenario_df = self.df.iloc[start:stop]
        scenario_x_emb = self.x_emb[start:stop]
        
        # Extract features for all alternatives in the scenario
        core_features = torch.tensor(scenario_df[self.core_vars].values, dtype=torch.float32)
        seg_features = torch.tensor(scenario_df[self.seg_vars].values, dtype=torch.float32)

        # Find the chosen alternative (the index within the scenario)
        choice = torch.tensor(scenario_df['match'].values.argmax(), dtype=torch.long)
        
        return {
            'core_features': core_features,
            'seg_features': seg_features,
            'x_emb': scenario_x_emb,
            'choice': choice
        }

def choice_collate_fn(batch):
    """
    Custom collate function to handle choice scenarios of variable lengths.

    This function takes a list of samples (dictionaries) from the GroupedChoiceDataset
    and pads them to create a single, uniform batch for the model.

    Args:
        batch (list[dict]): A list of dictionaries, where each dictionary
                            represents one choice scenario from __getitem__.

    Returns:
        dict: A dictionary containing batched and padded tensors for features,
              choices, and a mask to identify the padded elements.
    """
    # Find the maximum number of alternatives in this batch for padding
    max_len = max(item['core_features'].shape[0] for item in batch)

    # Prepare lists to hold padded tensors for each key
    core_features_list, seg_features_list, x_emb_list, masks_list, choices_list = [], [], [], [], []

    # Iterate through each sample (a choice scenario) in the batch
    for item in batch:
        # --- 1. Get the data for the current scenario ---
        core_feat = item['core_features']
        seg_feat = item['seg_features']
        x_emb = item['x_emb']
        
        # --- 2. Calculate how much padding is needed ---
        num_alternatives = core_feat.shape[0]
        padding_needed = max_len - num_alternatives

        # --- 3. Pad each feature tensor to max_len ---
        # `pad` takes (padding_left, padding_right, padding_top, padding_bottom)
        # We only need to pad the "bottom" (the alternatives dimension)
        core_features_list.append(torch.nn.functional.pad(core_feat, (0, 0, 0, padding_needed)))
        seg_features_list.append(torch.nn.functional.pad(seg_feat, (0, 0, 0, padding_needed)))
        
        # Ensure padding for embeddings uses a 0 of the correct long type
        x_emb_list.append(torch.nn.functional.pad(x_emb, (0, 0, 0, padding_needed), value=0))

        # --- 4. Create a boolean mask ---
        # `True` for real data, `False` for padding. The model uses this to ignore padded values.
        mask = torch.cat([torch.ones(num_alternatives, dtype=torch.bool), 
                          torch.zeros(padding_needed, dtype=torch.bool)])
        masks_list.append(mask)
        
        # --- 5. Collect the choice index ---
        choices_list.append(item['choice'])

    # --- 6. Stack the lists of tensors into a single batch tensor ---
    # The final dictionary that the model will receive in the training loop
    return {
        'core_features': torch.stack(core_features_list),
        'seg_features': torch.stack(seg_features_list),
        'x_emb': torch.stack(x_emb_list),
        'mask': torch.stack(masks_list),
        'choice': torch.stack(choices_list)
    }

class ChoiceDatasetByRow(Dataset):
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