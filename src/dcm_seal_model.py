# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
import pandas as pd

class DCM_SEAL(pl.LightningModule):
    def __init__(self, config: dict):
        """
        Initializes the DCM-SEAL model with a global embedding matrix architecture,
        correctly handling all specified architectural variations.

        Args:
            config (dict): A dictionary containing model hyperparameters.
                Expected keys:
                - 'n_latent_classes' (int): Number of latent classes (K).
                - 'n_alternatives' (int): Number of choice alternatives (J).
                - 'choice_mode' (str): 'heterogeneous' or 'homogeneous'.
                - 'embedding_mode' (str): 'shared' or 'class-specific'.
                - 'embedding_dims' (dict): An *ordered* dict mapping categorical var names
                  to their # of unique categories. e.g., {"purpose": 10, "hr": 24, ...}
                - 'segmentation_vars' (list[str]): Column names for segmentation.
                - 'core_vars' (list[str]): Column names for core utility variables.
                - 'segmentation_net_dims' (list[int]): Layer dimensions for the segmentation net.
                - 'learning_rate' (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters(config)

        # --- Store variable names and modes for easy access ---
        # The .get() method safely retrieves a value from the config dictionary.
        # The second argument is a default value to use if the key is not found.
        self.seg_vars = self.hparams.get("segmentation_vars", [])
        self.emb_vars = list(self.hparams.embedding_dims.keys()) # Order matters
        self.core_vars = self.hparams.get("core_vars", [])
        self.embedding_mode = self.hparams.get("embedding_mode", "shared")
        self.choice_mode = self.hparams.get("choice_mode", "heterogeneous")

        # --- Calculate Embedding Offsets and Total Size ---
        # Create a tensor of offsets for indexing into the global embedding matrix: i.e., E separator
        self.emb_offsets = torch.tensor([0] + list(self.hparams.embedding_dims.values())[:-1]).cumsum(dim=0)
        self.total_emb_categories = sum(self.hparams.embedding_dims.values()) #break down E to get Z

        # --- MODEL DEFINITION BASED ON NUMBER OF LATENT CLASSES ---
        if self.hparams.n_latent_classes > 1:
            # --- PATH 1: LATENT CLASS MODEL (K > 1) ---
            print(f"Initializing a Latent Class Model with K={self.hparams.n_latent_classes} classes.")

            # 1. Segmentation Component (Required for K > 1)
            if len(self.seg_vars) == 0:
                raise ValueError(
                    "For a latent class model (n_latent_classes > 1), "
                    "'segmentation_vars' must be provided in the config."
                )
            seg_layers = []
            for i in range(len(self.hparams.segmentation_net_dims) - 1):
                seg_layers.append(nn.Linear(self.hparams.segmentation_net_dims[i], self.hparams.segmentation_net_dims[i+1]))
                seg_layers.append(nn.ReLU())
            self.segmentation_net = nn.Sequential(*seg_layers)

            # 2. Global Embedding Layer(s) - CONDITIONAL ON MODE
            if self.embedding_mode == 'class-specific':
                print("Initializing with CLASS-SPECIFIC global embedding matrices.")
                # nn.Embedding module creates an (Z,J) learnable matrix expecting an index input [0,Z-1] and output z-th row with len J
                self.embedding_layers = nn.ModuleList([
                    nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives) # (Z,J)
                    for _ in range(self.hparams.n_latent_classes)
                ])
            else: # 'shared' mode
                self.embedding_layers = nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives)

            # 3. Coefficients (all have a class dimension)
            self.core_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.core_vars))) #(K,C)
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.emb_vars))) #(K,E)

            # 4. ASCs (have a class dimension)
            if self.choice_mode == 'heterogeneous':
                print("Initializing with Alternative Specific Constants (ASCs).")
                self.asc = nn.Parameter(torch.randn(self.hparams.n_latent_classes, self.hparams.n_alternatives - 1)) #(K,J-1)

        else:
            # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
            print("Initializing a Single Class Model (K=1) without Segmentation Network.")

            # 1. Segmentation Component is SKIPPED

            # 2. Global Embedding Layer
            if len(self.emb_vars) > 0:
                self.embedding_layers = nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives)

            # 3. Coefficients (no class dimension)
            if len(self.core_vars) > 0:
                self.core_betas = nn.Parameter(torch.randn(len(self.core_vars))) #(C,)
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(len(self.emb_vars))) #(E,)

            # 4. ASCs (no class dimension)
            if self.choice_mode == 'heterogeneous':
                self.asc = nn.Parameter(torch.randn(self.hparams.n_alternatives - 1)) #(J-1,)


    def forward(self, batch: dict[str, torch.Tensor]):
        """
        Implements the forward pass with dedicated logic for single-class (K=1) and multi-class (K>1) models.
        Note that J dimension disappears in the return because we are dealing with long input data format (each row->specific alt.)
        The next training_step will receive the (B,) output tensor from this forward() pass.
        """
        # Ensure the offset tensor is on the same device as the model (e.g., GPU)
        self.emb_offsets = self.emb_offsets.to(self.device)

        if self.hparams.n_latent_classes > 1:
            # --- PATH 1: LATENT CLASS MODEL (K > 1) ---

            # 1. Segmentation Probabilities
            x_seg = torch.cat([batch[var].unsqueeze(1) for var in self.seg_vars], dim=1).float() # dim: (B, S)
            class_logits = self.segmentation_net(x_seg) # dim: (B, K)
            class_probs = F.softmax(class_logits, dim=1) # dim: (B, K)

            # 2. Core Utility
            '''
            Explanation of einsum 'kc,bc->bk' with latent 'k'lasses, 'c'ore vars, and 'b'atch ids:
                The dimension of self.core_betas is (K,C), x_core is (B,C), the einsum between should output a matrix with (B,K)
                This einsum definition represents a batch matrix multpilication across the c dimension (i.e., not appearing in ->bk)
                The output matrix's i-th row and j-th column stores the dot product of the latter's i-th row (b) and the foremr's j-th ROW (k)
            '''
            utility_core_by_class = 0.0
            if len(self.core_vars) > 0:
                x_core = torch.cat([batch[var].unsqueeze(1) for var in self.core_vars], dim=1).float() # concat C (B,1)'s-> dim: (B, C)
                utility_core_by_class = torch.einsum('kc,bc->bk', self.core_betas, x_core) # dim: (B, K)

            # 3. Embedding Utility
            utility_from_embeddings_per_class = 0.0
            if len(self.emb_vars) > 0:
                # Add offsets to create indices for the global embedding matrix
                x_emb_with_offsets = batch['x_emb'] + self.emb_offsets # a matrix of integer indices; dim: (B, E)
                positive_betas = F.softplus(self.embedding_betas) # dim: softplus((K,E))->still (K,E)

                # This will hold the utility from embeddings for each class, shape (B, J)
                '''
                Explanation of an element in raw_embs definition self.embedding_layers[k][name](x_emb[name]):
                    layer=self.embedding_layers[k]: the k-th embedding layer (assoc. with LC k) of our module list defined above
                    x_emb_with_offsets: A tensor with integer offsetted index for emb var's dim: (B,E)
                    The object 'layer' itself is a function including learnable weight matrix that USES x_emb_with_offsets AS ITS INPUT.
                    Ultimately, it is equivalent to do (B, E) times of an index-based lookup in (Z,J) weight matrix to return row (len J vector)
                '''
                class_embedding_utilities = []
                for k in range(self.hparams.n_latent_classes):
                    # Select the correct embedding layer for the mode and class
                    layer = self.embedding_layers[k] if self.embedding_mode == 'class-specific' else self.embedding_layers

                    # Look up all E embedding vectors for each person
                    raw_embs = layer(x_emb_with_offsets) # dim: (B, E, J)
                    norm_embs = F.normalize(raw_embs, p=2, dim=2) # dim: (B, E, J)

                    # Apply class-specific betas for each of the E variables
                    betas_k = positive_betas[k].unsqueeze(0).unsqueeze(2) # dim: (1, E, 1)

                    # Sum the weighted utilities from each of the E variables
                    utility_k = torch.sum(norm_embs * betas_k, dim=1) # dim: (B, J)
                    class_embedding_utilities.append(utility_k) # a list of length K where each element being (B,J) tensor

                # Stack the results to get the utility for all classes
                utility_emb_by_class = torch.stack(class_embedding_utilities, dim=1) # dim: (B, K, J): a big Excel with B sheets

                # Gather the utility for the specific alternative in each row
                selector_index = batch['alt'].long().unsqueeze(1).unsqueeze(2) #alt. index like tensor ([[0],[2],...]); dim: (B,1,1)
                selector_index = selector_index.expand(-1, self.hparams.n_latent_classes, -1) # duplicate data K times; dim: (B, K, 1)
                utility_from_embeddings_per_class = utility_emb_by_class.gather(2, selector_index).squeeze(2) # get (b,k) utils for given j (=dim 2)
                #Converts the retrieved (B,K,1) via gather to (B,K) after squeeze; represents each line of long (J-implied), K-expanded input df

            # 4. Combine and Add ASCs
            class_specific_utility = utility_core_by_class + utility_from_embeddings_per_class # dim: (B, K)
            if self.choice_mode == 'heterogeneous':
                zeros = torch.zeros(self.hparams.n_latent_classes, 1, device=self.device) # dim: (K, 1)
                full_asc = torch.cat([self.asc, zeros], dim=1) # dim: (K,(J-1)+1)->(K, J)
                class_specific_utility += full_asc[:, batch['alt'].long()].T # []: select alt j's ASC (len K) B times->+=(K, B).T

            # 5. Final Weighted Utility
            final_utility = torch.sum(class_probs * class_specific_utility, dim=1) # elem-wise mult two (B,K)s summed over K ->(B,)
            return final_utility

        else:
            # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
            final_utility = 0.0

            # 1. Core Utility
            if len(self.core_vars) > 0:
                x_core = torch.cat([batch[var].unsqueeze(1) for var in self.core_vars], dim=1).float() # dim: (B, C)
                final_utility += torch.einsum('c,bc->b', self.core_betas, x_core) # dim: (B,)

            # 2. Embedding Utility
            if len(self.emb_vars) > 0:
                x_emb_with_offsets = batch['x_emb'] + self.emb_offsets # dim: (B, E)
                positive_betas = F.softplus(self.embedding_betas) # dim: (E,)

                raw_embs = self.embedding_layers(x_emb_with_offsets) # dim: (B, E, J)
                norm_embs = F.normalize(raw_embs, p=2, dim=2) # dim: (B, E, J)

                # Apply betas and sum across the E variables: sum((B, E, J)*(1, E, 1),E dim)
                utility_from_embeddings = torch.sum(norm_embs * positive_betas.unsqueeze(0).unsqueeze(2), dim=1) # dim: (B, J)

                # Gather the utility for the specific alternative in the row
                alt_indices = batch['alt'].long().unsqueeze(1) # dim: (B, 1)
                final_utility += utility_from_embeddings.gather(1, alt_indices).squeeze(1) # dim: (B,)

            # 3. ASC Utility
            if self.choice_mode == 'heterogeneous' and self.hparams.n_alternatives > 1:
                zeros = torch.zeros(1, device=self.device) # dim: (1,)
                full_asc = torch.cat([self.asc, zeros]) # dim: (J,)
                final_utility += full_asc[batch['alt'].long()] # dim: (B,)

            return final_utility

    def get_embedding_weights(self,inds,cols, conv_file_path: str = None):
        """
        Extracts the full, trained embedding weight matrix (Z, J) from the
        embedding layer(s).
    
        Args:
            conv_file_path (str, optional): Path to the dfConv.csv file. If provided,
                the output DataFrame will have a detailed MultiIndex with original
                category labels for the rows. Defaults to None.
    
        Returns:
            pandas.DataFrame or dict:
            - If 'shared' mode: A single DataFrame of shape (Z, J).
            - If 'class-specific' mode: A dictionary of DataFrames, one for each class k.
        """

        # Set model to evaluation mode and disable gradients
        self.eval()
        with torch.no_grad():
            if self.hparams.n_latent_classes > 1 and self.embedding_mode == 'class-specific':
                # --- PATH 1: CLASS-SPECIFIC MODE ---
                class_weights = {}
                for k in range(self.hparams.n_latent_classes):
                    # Access the .weight attribute of the embedding layer for class k
                    weight_matrix = self.embedding_layers[k].weight.cpu().numpy()
                    class_weights[k] = pd.DataFrame(weight_matrix, index=inds, columns=cols)
                return class_weights
    
            else:
                # --- PATH 2: SHARED or K=1 MODE ---
                # Access the .weight attribute of the single embedding layer
                weight_matrix = self.embedding_layers.weight.cpu().numpy()
                return pd.DataFrame(weight_matrix, index=inds, columns=cols)

    def training_step(self, batch, batch_idx):
        """
        Defines a single step of training.
        
        This method will compute the loss for a batch and return it.
        PyTorch Lightning handles the backpropagation and optimizer steps automatically.
        """
        # To be implemented in our next step...
        pass

    def configure_optimizers(self):
        """
        Sets up the optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == '__main__':
    pass
