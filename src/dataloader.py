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