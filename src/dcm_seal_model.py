# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DCM_SEAL(pl.LightningModule):
    def __init__(self, config: dict):
        """
        Initializes the full DCM-SEAL model, capable of handling all specified architectural variations.

        Args:
            config (dict): A dictionary containing model hyperparameters.
                Expected keys:
                - 'n_latent_classes' (int): Number of latent classes (K).
                - 'n_alternatives' (int): Number of choice alternatives (J).
                - 'choice_mode' (str): 'heterogeneous' or 'homogeneous'.
                - 'embedding_mode' (str): 'shared' or 'class-specific'.
                - 'segmentation_vars' (list[str]): Column names for segmentation.
                - 'embedding_dims' (dict): Maps categorical var names to their # of unique categories. e.g., {"purpose": 10, ...}
                - 'core_vars' (list[str]): Column names for core utility variables.
                - 'segmentation_net_dims' (list[int]): Layer dimensions for the segmentation net.
                - 'learning_rate' (float): Learning rate for the optimizer.
        
        Dimensions:
            batch_size: B, n_latent_classes: K, n_alternatives: J, n_seg_vars: S, n_core_vars: C, n_emb_vars: E
        """
        super().__init__()
        self.save_hyperparameters(config)

        # --- Store variable names and modes; .get() args: (name, default) ---
        self.seg_vars = self.hparams.get("segmentation_vars", [])
        self.emb_vars = list(self.hparams.embedding_dims.keys())
        self.core_vars = self.hparams.get("core_vars", [])
        self.embedding_mode = self.hparams.get("embedding_mode", "shared")
        self.choice_mode = self.hparams.get("choice_mode", "heterogeneous")

        # --- MODEL DEFINITION BASED ON NUMBER OF LATENT CLASSES ---
        if self.hparams.n_latent_classes > 1:
            # --- PATH 1: LATENT CLASS MODEL (K > 1) ---
            print(f"Initializing a Latent Class Model with K={self.hparams.n_latent_classes} classes.")

            # 1. Segmentation Component
            if len(self.seg_vars) == 0:
                raise ValueError(
                    "For a latent class model (n_latent_classes > 1), "
                    "'segmentation_vars' must be provided in the config."
                )

            print("Initializing with a segmentation network.")
            seg_layers = []
            for i in range(len(self.hparams.segmentation_net_dims) - 1):
                seg_layers.append(nn.Linear(self.hparams.segmentation_net_dims[i], self.hparams.segmentation_net_dims[i+1]))
                seg_layers.append(nn.ReLU())
            self.segmentation_net = nn.Sequential(*seg_layers)

            # 2. Embedding Layers (Conditional on mode)
            if self.embedding_mode == 'class-specific':
                print("Initializing with CLASS-SPECIFIC embedding weight matrices.")
                self.embedding_layers = nn.ModuleList([
                    nn.ModuleDict({
                        name: nn.Embedding(num_embeddings, self.hparams.n_alternatives)
                        for name, num_embeddings in self.hparams.embedding_dims.items()
                    }) for _ in range(self.hparams.n_latent_classes)
                ])
            else: # 'shared'
                print("Initializing with SHARED embedding weight matrices.")
                self.embedding_layers = nn.ModuleDict({
                    name: nn.Embedding(num_embeddings, self.hparams.n_alternatives)
                    for name, num_embeddings in self.hparams.embedding_dims.items()
                })

            # 3. Coefficients (all have a class dimension)
            self.core_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.core_vars))) #dim: (K, C)
            self.embedding_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.emb_vars))) #dim: (K, E)

            # 4. ASCs (have a class dimension)
            if self.choice_mode == 'heterogeneous':
                print("Initializing with Alternative Specific Constants (ASCs).")
                self.asc = nn.Parameter(torch.randn(self.hparams.n_latent_classes, self.hparams.n_alternatives - 1)) #dim: (K, J-1)

        else:
            # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
            print("Initializing a Single Class Model (K=1). Segmentation network is disabled.")

            # 1. Segmentation Component is SKIPPED

            # 2. Embedding Layers (only one shared set is needed)
            if len(self.emb_vars) > 0:
                self.embedding_layers = nn.ModuleDict({
                    name: nn.Embedding(num_embeddings, self.hparams.n_alternatives)
                    for name, num_embeddings in self.hparams.embedding_dims.items()
                })

            # 3. Coefficients (no class dimension needed)
            if len(self.core_vars) > 0:
                self.core_betas = nn.Parameter(torch.randn(len(self.core_vars))) #dim: C
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(len(self.emb_vars))) #dim: E

            # 4. ASCs (no class dimension needed)
            if self.choice_mode == 'heterogeneous':
                self.asc = nn.Parameter(torch.randn(self.hparams.n_alternatives - 1))

    def forward(self, batch: dict[str, torch.Tensor]):
        """
        Implements the forward pass with dedicated logic for single-class (K=1) and multi-class (K>1) models.
        Note that J dimension disappears in the return because we are dealing with long input data format (each row->specific alt.)
        The next training_step will receive the (B,) output tensor from this forward() pass.
        """
        if self.hparams.n_latent_classes > 1:
            # --- PATH 1: LATENT CLASS MODEL (K > 1) ---

            # 1. Segmentation Probabilities
            x_seg = torch.cat([batch[var].unsqueeze(1) for var in self.seg_vars], dim=1).float() # dim: (B,S)
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
                x_core = torch.cat([batch[var].unsqueeze(1) for var in self.core_vars], dim=1).float() # dim: (B,C)
                utility_core_by_class = torch.einsum('kc,bc->bk', self.core_betas, x_core) # dim: (B,K); see explanation

            # 3. Embedding Utility
            utility_from_embeddings_per_class = 0.0
            if len(self.emb_vars) > 0:
                x_emb = {key: batch[key].long() for key in self.emb_vars} # len E dict: {'purpose':tensor([...],dtype= int64):len B, 'hr':...}
                positive_betas = F.softplus(self.embedding_betas) # dim: softplus((K,E))->still (K,E)

                if self.embedding_mode == 'class-specific':
                    '''
                    Explanation of an element in raw_embs definition self.embedding_layers[k][name](x_emb[name]):
                        self.embedding_layers[k]: the k-th dictionary of embedding layers
                        .[name]: get specific weight matrix for the name e.g., 'purpose'; size (purpose_cardinality=P,J) learnable weight matrix
                        x_emb[name]: A tensor with integer category for the name e.g., 'purpose'; len B tensor([1,4,2,1...],dypte=int64)
                        The object self.embedding_layers[k][name] is a function that USES the emb weight matrix, x_emb[name] AS ITS INPUT.
                        Ultimately, it is equivalent to do a matrix multiplication of binary-populated x_emb[name] (B,P) and the weight matrix (P,J)
                    '''
                    class_utilities = []
                    for k in range(self.hparams.n_latent_classes): #for each latent class
                        raw_embs = [self.embedding_layers[k][name](x_emb[name]) for name in self.emb_vars] # see above
                        norm_embs = [F.normalize(emb, p=2, dim=1) for emb in raw_embs] # same as raw_embs: list of E tensors, each with (B,J)
                        stacked_embs = torch.stack(norm_embs, dim=2) #change the dim to (B, J, E) by staking
                        class_k_utility = torch.sum(stacked_embs * positive_betas[k], dim=2) #sum ((B,J,E) * (E),...)=sum((B,J,E),dim=2)->(B,J)
                        class_utilities.append(class_k_utility) #after the loop, len K list of (B,J) tensors
                    utility_emb_by_class = torch.stack(class_utilities, dim=1) # (B,K,J)
                else: # 'shared'
                    raw_embs = [self.embedding_layers[name](x_emb[name]) for name in self.emb_vars] # list of E tensors, each with (B,J)
                    norm_embs = [F.normalize(emb, p=2, dim=1) for emb in raw_embs] # same
                    stacked_embs = torch.stack(norm_embs, dim=2).unsqueeze(1) # (B, 1, J, E) after unsqueezing the stacked (B,J,E)
                    betas = positive_betas.unsqueeze(0).unsqueeze(2) # (1, K, 1, E) after double-unsqueezing positive_beats (K,E)
                    utility_emb_by_class = torch.sum(stacked_embs * betas, dim=3) #sum(broadcasted (B,K,J,E),dim=3)-> (B,K,J)

                selector_index = batch['alt'].long().unsqueeze(1).unsqueeze(2) # alt. index like tensor([[[0],[2],...]]); dim: (B,1,1)
                selector_index = selector_index.expand(-1, self.hparams.n_latent_classes, -1) # duplicate data to make dim: (B,K,1)
                utility_from_embeddings_per_class = utility_emb_by_class.gather(2, selector_index).squeeze(2) #get (b,k) utils for given j (=dim 2)
                #now utility_from_embeddings_per_class has a dimension (B,K) after squeeze(2)

            # 4. Combine and Add ASCs
            class_specific_utility = utility_core_by_class + utility_from_embeddings_per_class # dim: (B,K)
            if self.choice_mode == 'heterogeneous' and self.hparams.n_alternatives > 1:
                zeros = torch.zeros(self.hparams.n_latent_classes, 1, device=self.device) # dim: (K,1)
                full_asc = torch.cat([self.asc, zeros], dim=1) # dim: (K,(J-1)+1)-> (K,J)
                class_specific_utility += full_asc[:, batch['alt'].long()].T # select appropriate j's from full_asc for all b-> +=(K,B).T

            # 5. Final Weighted Utility
            final_utility = torch.sum(class_probs * class_specific_utility, dim=1) #elem-wise mult (B,K) summed over K ->dim B
            return final_utility

        else:
            # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
            final_utility = 0.0

            # 1. Core Utility
            if len(self.core_vars) > 0:
                x_core = torch.cat([batch[var].unsqueeze(1) for var in self.core_vars], dim=1).float() # dim: (B,C)
                final_utility += torch.einsum('c,bc->b', self.core_betas, x_core) # dim: B

            # 2. Embedding Utility
            if len(self.emb_vars) > 0:
                x_emb = {key: batch[key].long() for key in self.emb_vars} # len E dict of len B tensors
                positive_betas = F.softplus(self.embedding_betas) # dim: E
                raw_embs = [self.embedding_layers[name](x_emb[name]) for name in self.emb_vars] # list of E tensors, each with (B,J)
                norm_embs = [F.normalize(emb, p=2, dim=1) for emb in raw_embs] #same
                stacked_embs = torch.stack(norm_embs, dim=2) #dim: (B, J, E)
                utility_from_embeddings = torch.sum(stacked_embs * positive_betas, dim=2) #dim: (B, J)
                alt_indices = batch['alt'].long().unsqueeze(1) #dim: (B, 1)
                final_utility += utility_from_embeddings.gather(1, alt_indices).squeeze(1) #get b utils for given j (=dim 1)
                #now final_utility has a dimension B after squeeze(1)

            # 3. ASC Utility
            if self.choice_mode == 'heterogeneous' and self.hparams.n_alternatives > 1:
                zeros = torch.zeros(1, device=self.device) # dim: 1
                full_asc = torch.cat([self.asc, zeros]) # dim: (J-1)+1-> J
                final_utility += full_asc[batch['alt'].long()] # select appropriate j's from full_asc for all b-> dim B
            return final_utility

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
    config = {
        "n_latent_classes": 4,
        "n_alternatives": 3,
        "choice_mode": "heterogeneous",
        "embedding_mode": "class-specific",
        "segmentation_vars": [
            "age_18to24", "age_25to34", # etc. for all one-hot encoded age groups
            "income_35K60K", "income_60K80K" # etc. for all income groups
        ],
        "embedding_dims": {
            "purpose": 10,  # 10 unique categories for trip purpose
            "hr": 24,       # 24 unique categories for hour of day
            "worktype": 5   # 5 unique categories for work type
        },
        "core_vars": ["cost", "iv", "ov", "nTrans"],
        "segmentation_net_dims": [64, 32, 16, 4],
        "learning_rate": 0.001
    }