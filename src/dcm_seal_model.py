# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict #needed for config's embedding dims and offset calculation
import pandas as pd

class DCM_SEAL(pl.LightningModule):
    def __init__(self, config: dict):
        """
        Initializes the DCM-SEAL model with a global embedding matrix architecture,
        correctly handling all specified architectural variations.

        Args:
            config (dict): A dictionary containing model hyperparameters; defined in main.py
                Expected keys:
                - 'n_latent_classes' (int): Number of latent classes.
                - 'n_alternatives' (int): Number of maximum choice alternatives.
                - 'choice_mode' (str): 'heterogeneous' or 'homogeneous'.
                - 'embedding_mode' (str): 'shared' or 'class-specific'.
                - 'embedding_dims' (dict): An *ordered* dict mapping categorical var names
                  to their # of unique categories. e.g., {"purpose": 10, "hr": 24, ...}
                - 'segmentation_vars' (list[str]): Column names for segmentation variables.
                - 'core_vars' (list[str]): Column names for core utility variables.
                - 'segmentation_net_dims' (list[int]): The numbers of hidden nodes for the segmentation NW layers.
                - 'segmentation_dropout_rate' (float): Dropout rate for the segmentation net.
                - 'weight_decay_segmentation' (float): Weight decay for the segmentation net.
                - 'weight_decay_embedding' (float): Weight decay for the embedding layers.
                - 'learning_rate' (float): Learning rate for the optimizer.

        Dimensions Guide:
            n_unique_chids: N, max(n_alts): J, batch_size (# of long input rows): B; NJ=B for ideal input data
            n_latent_classes: K, n_seg_vars: S, n_core_vars: C, n_emb_vars: E, n_sum_of_categories_across_emb_vars: Z
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
        # e.g., length E 1D tensor [0, 2, 6, 10, 12, 15, 18, 20]
        self.total_emb_categories = sum(self.hparams.embedding_dims.values()) #break down E to get Z

        # --- MODEL DEFINITION BASED ON NUMBER OF LATENT CLASSES ---
        if self.hparams.n_latent_classes > 1:
            # --- PATH 1: LATENT CLASS MODEL (K > 1) ---
            print(f"Initializing a Latent Class Model with K={self.hparams.n_latent_classes} classes.")

            # 1. Segmentation Component (Required for K > 1), with Dropout
            if len(self.seg_vars) == 0:
                raise ValueError(
                    "For a latent class model (n_latent_classes > 1), "
                    "'segmentation_vars' must be provided in the config."
                )
            dropout_rate = self.hparams.get("segmentation_dropout_rate", 0.0)
            print(f"Initializing segmentation network with dropout rate: {dropout_rate}")
            
            seg_layers = []
            layer_dims = self.hparams.segmentation_net_dims
            for i in range(len(layer_dims) - 1):
                seg_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
                # Don't add ReLU or Dropout after the final layer that produces logits
                if i < len(layer_dims) - 2:
                    seg_layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        seg_layers.append(nn.Dropout(p=dropout_rate))
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
            if len(self.core_vars) > 0:
                self.core_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.core_vars))) # (K,C)
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(self.hparams.n_latent_classes, len(self.emb_vars))) # (K,E)

            # 4. ASCs (have a class dimension)
            if self.choice_mode == 'heterogeneous':
                print("Initializing with Alternative Specific Constants (ASCs).")
                self.asc = nn.Parameter(torch.randn(self.hparams.n_latent_classes, self.hparams.n_alternatives - 1)) # (K,J-1)

        else:
            # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
            print("Initializing a Single Class Model (K=1) without Segmentation Network.")

            # 1. Segmentation Component is SKIPPED

            # 2. Global Embedding Layer
            if len(self.emb_vars) > 0:
                self.embedding_layers = nn.Embedding(self.total_emb_categories, self.hparams.n_alternatives) # (Z,J)

            # 3. Coefficients (no class dimension)
            if len(self.core_vars) > 0:
                self.core_betas = nn.Parameter(torch.randn(len(self.core_vars))) #(C,)
            if len(self.emb_vars) > 0:
                self.embedding_betas = nn.Parameter(torch.randn(len(self.emb_vars))) #(E,)

            # 4. ASCs (no class dimension)
            if self.choice_mode == 'heterogeneous':
                self.asc = nn.Parameter(torch.randn(self.hparams.n_alternatives - 1)) #(J-1,)
        try:
            self.project_embedding_rows_unit_l2()
        except Exception:
            pass  # Safe to skip; on_fit_start will handle it

    def forward(self, batch: dict[str, torch.Tensor]):
        """
        This method is aligned with a DataLoader that separates per-alternative data
        (e.g., core_features) from per-scenario data (e.g., seg_features, x_emb).
        Legend:
            B: batch size (#chids in batch)
            J: #alternatives (<= config["n_alternatives"])
            C: #core variables in this run
            E: #embedding variables (count, not total categories)
            Z: total categories across all embedding variables (sum of cardinalities)
            K: #latent classes
        Args:
            batch (dict): A dictionary from the DataLoader with efficient shapes:
                          'core_features': (B, J, C)
                          'seg_features':  (B, S)
                          'x_emb':         (B, E)
                          'mask':          (B, J)  - Boolean; 1 for real alternatives.
                          'choice':        (B,)
    
        Returns:
            torch.Tensor: A tensor of final utilities of shape (B, J).
        """
        # Get the device from an input tensor to ensure all new tensors are on the same device (e.g., GPU)

        # --- PATH 1: LATENT CLASS MODEL (K > 1) ---
        if self.hparams.n_latent_classes > 1:
            # 1. --- Segmentation Probabilities ---
            x_seg = batch['seg_features']  # Shape: (B, S)
            class_logits = self.segmentation_net(x_seg)
            class_probs = F.softmax(class_logits, dim=1)  # Shape: (B, K)

            # 2. --- Core Utility (per-alternative) ---
            '''explanation of einsum: kc, bjc->bjk (RHS can be any combination using b j k but we deliberately do bjk)
            einsum: broadcasted dot products are done with vectors of the shared letter (dimension), for our case: c
            b0 k1 (C=3 J=2) example: [beta_k1c0, beta_k1c1, beta_k1c2] dot [[x_b0j0c0, x_b0j0c1, x_b0j0c2]:for alt j=0,
            [x_b0j1c0, x_b0j1c1, x_b0j1c2]:for alt j=1]-> [CoreUtil_j0, CoreUtil_j1] this happens for every b and k ->(B,J,K)
            
            '''
            utility_core_by_class = torch.zeros(1, device=self.device)
            if self.core_vars:
                x_core = batch['core_features']  # Shape: (B, J, C)
                utility_core_by_class = torch.einsum('kc,bjc->bjk', self.core_betas, x_core)

            # 3. --- Embedding Utility (per-alternative) ---
            '''
            Explanation of an element in raw_embs definition self.embedding_layers[k][name](x_emb[name]):
                layer=self.embedding_layers[k]: the k-th embedding layer (assoc. with LC k) of our module list defined above
                x_emb_with_offsets: A tensor with integer offsetted index for emb var's dim: (B,E)
                The object 'layer' itself is a function including learnable weight matrix that USES x_emb_with_offsets AS ITS INPUT.
                Ultimately, it is equivalent to do B times of input (E,Z) matmul (Z,J) weight matrix to return (B,E,J)
            '''
            utility_from_embeddings_per_class = torch.zeros(1, device=self.device)
            if self.emb_vars: # x_emb_with_offsets:  a matrix of integer indices (B,E) + (E,)
                x_emb_with_offsets = batch['x_emb'] + self.emb_offsets.to(self.device) # broadcast: (B, E)
                positive_betas = F.softplus(self.embedding_betas) # Shape: (K, E)
                class_embedding_utilities = []
                for k in range(self.hparams.n_latent_classes):
                    layer = self.embedding_layers[k] if self.embedding_mode == "class-specific" else self.embedding_layers
                    raw_embs = layer(x_emb_with_offsets) #dim: (B, E, J)
                    norm_embs = raw_embs # assign just for a formality; rows already unit L2 via projector (see on_ methods below)
                    # norm_embs = F.normalize (raw_embs, p=2, dim=2) # legacy code for lightning; no need to define l2, on_methods but slow
                    betas_k = positive_betas[k].unsqueeze(0).unsqueeze(2)  # class-specific betas (1, E, 1)
                    utility_k = torch.sum(norm_embs * betas_k, dim=1) # (B, E, J) * (1, E, 1), sum over E -> (B, J) 
                    class_embedding_utilities.append(utility_k) # a list of length K where each element being (B, J) tensor
                # Stack utilities for all K classes -> (B, K, J)
                utility_emb_stacked = torch.stack(class_embedding_utilities, dim=1)
                # Permute to align with core utility shape -> (B, J, K)
                utility_from_embeddings_per_class = utility_emb_stacked.permute(0, 2, 1)

            # 4. --- Combine Utilities and Add ASCs ---
            class_specific_utility = utility_core_by_class + utility_from_embeddings_per_class

            if self.choice_mode == 'heterogeneous':
                zeros = torch.zeros(self.hparams.n_latent_classes, 1, device=self.device)
                full_asc = torch.cat([self.asc, zeros], dim=1)
                asc_reshaped = full_asc.T.unsqueeze(0)
                class_specific_utility += asc_reshaped

            # 5. --- Final Weighted Utility ---
            final_utility = torch.sum(class_specific_utility * class_probs.unsqueeze(1), dim=2)

        # --- PATH 2: SINGLE CLASS MODEL (K = 1) ---
        else:
            # 1. --- Core Utility (per-alternative) ---
            utility_core = torch.zeros(1, device=self.device)
            if self.core_vars:
                x_core = batch['core_features'] # (B, J, C)
                utility_core = torch.einsum('c,bjc->bj', self.core_betas, x_core)

            # 2. --- Embedding Utility (per-alternative) ---
            utility_emb = torch.zeros(1, device=self.device)
            if self.emb_vars:
                x_emb_with_offsets = batch['x_emb'] + self.emb_offsets.to(self.device)
                positive_betas = F.softplus(self.embedding_betas) # Shape: (E,)
                raw_embs = self.embedding_layers(x_emb_with_offsets) # Shape: (B, E, J)
                norm_embs = raw_embs  #already unit L2
                # norm_embs = F.normalize (raw_embs, p=2, dim=2) # legacy
                utility_emb = torch.sum(norm_embs * positive_betas.unsqueeze(0).unsqueeze(2), dim=1)

            # 3. --- ASC Utility (per-alternative) ---
            utility_asc = torch.zeros(1, device=self.device)
            if self.choice_mode == 'heterogeneous':
                zeros = torch.zeros(1, device=self.device)
                full_asc = torch.cat([self.asc.squeeze(0), zeros])
                utility_asc = full_asc.unsqueeze(0)

            # 4. --- Final Utility ---
            # Note: In the K=1 case, embedding utility is per-alternative, so no unsqueeze is needed.
            final_utility = utility_core + utility_emb + utility_asc

        # --- Final Step (CRITICAL): Apply Masking ---
        final_utility = final_utility.masked_fill(~batch['mask'], -torch.inf)
        return final_utility

    def project_embedding_rows_unit_l2(self):
        """
        Enforce row-wise unit L2 norm on the embedding weight matrix/matrices.

        Shapes:
          - Shared embedding:           W {in} R^(Z, J)
          - Class-specific embeddings:  {W_k} with each W_k {in} R^(Z, J), k=1..K
        We normalize along dim=1 (rows), so ||W[z, :]||2 = 1 for all z.
        """
        # If no embedding variables in this run, nothing to do
        if len(self.emb_vars) == 0:
            return

        with torch.no_grad():
            if self.embedding_mode == "class-specific":
                for layer in self.embedding_layers:          # layer: nn.Embedding(Z, J)
                    W = layer.weight                          # (Z, J)
                    norms = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    W.div_(norms)                             # in-place row-wise normalize
            else:
                W = self.embedding_layers.weight              # (Z, J)
                norms = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                W.div_(norms)

    # Lightning calls on_train_batch_end after the backward + optimizer step (i.e., reserved names),
    # so these two methods are substituting the classic “projected gradient” placement.
    def on_fit_start(self):
        # Ensure a normalized starting point
        self.project_embedding_rows_unit_l2()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # After optimizer.step() has run, project weights back to the unit sphere
        self.project_embedding_rows_unit_l2()

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer with different weight_decay values for
        different parameter groups.
        """
        # Get the regularization hyperparameters, with defaults
        decay_seg = self.hparams.get("weight_decay_segmentation", 1e-2) # Default: Strong decay
        decay_emb = self.hparams.get("weight_decay_embedding", 1e-4) # Default: Moderate decay

        # --- Create Parameter Groups ---

        # Group 1: The segmentation network parameters
        # This group only exists for the latent class model (K > 1)
        param_groups = []
        if self.hparams.n_latent_classes > 1:
            param_groups.append({
                'params': self.segmentation_net.parameters(),
                'weight_decay': decay_seg
            })

        # Group 2: The embedding layer weights
        if len(self.emb_vars) > 0:
            param_groups.append({
                'params': self.embedding_layers.parameters(),
                'weight_decay': decay_emb
            })

        # Group 3: All other betas and ASCs, with NO weight decay.
        # We collect them in a list first to handle cases where they might not exist.
        no_decay_params = []
        if len(self.core_vars) > 0:
            no_decay_params.append(self.core_betas)
        if len(self.emb_vars) > 0:
            no_decay_params.append(self.embedding_betas)
        if self.choice_mode == 'heterogeneous' and hasattr(self, 'asc'):
            no_decay_params.append(self.asc)

        if no_decay_params: # Only add the group if it's not empty
            param_groups.append({
                'params': no_decay_params,
                'weight_decay': 0.0
            })

        # --- Initialize the Optimizer ---
        optimizer = torch.optim.AdamW(param_groups,lr=self.hparams.learning_rate)
        return optimizer

    def _calculate_loss(self, batch):
        """
        Helper function to calculate loss.
        """
        # 1. Get Utilities from the forward pass
        # The shape of utilities is now (BatchSize, NumAlternatives)
        utilities = self.forward(batch)
    
        # 2. Get the index of the chosen alternative for each scenario in the batch
        # This comes directly from the dataloader.
        choice_indices = batch['choice'] # Shape: (BatchSize,) indices of chosen alt (0 to J-1)
    
        # 3. Calculate cross-entropy loss
        # This is the standard negative log-likelihood for choice models.
        # It implicitly applies a softmax to the utilities.
        loss = F.cross_entropy(utilities, choice_indices)
        
        # Calculate accuracy
        preds = torch.argmax(utilities, dim=1)
        acc = (preds == choice_indices).float().mean()
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        loss, acc = self._calculate_loss(batch)
        # Log metrics for the epoch.
        self.log_dict(
            {'train_loss': loss, 'train_acc': acc},
            on_step=False, # We only need the final epoch value
            on_epoch=True,
            prog_bar=False
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        loss, acc = self._calculate_loss(batch)
        # The final, correct values will be logged at the end of the epoch.
        self.log_dict(
            {'val_loss': loss, 'val_acc': acc},
            on_epoch=True,
            prog_bar=True # It's often useful to see val_loss in the bar
        )
    
    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.
        """
        loss, acc = self._calculate_loss(batch)
        self.log_dict(
            {'test_loss': loss, 'test_acc': acc},
            on_epoch=True,
            prog_bar=True
        )


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
# DCM-SEAL has been configured.

#%% deprecated, "long" format assumed
def forward_long_deprecated(self, batch: dict[str, torch.Tensor]):
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
        # x_seg = torch.cat([batch[var].unsqueeze(1) for var in self.seg_vars], dim=1).float() # dim: (B, S)
        x_seg = batch['seg_features'] # dim: (B, S); replacing the above line with dataloader.py
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
        if self.choice_mode == 'heterogeneous':
            zeros = torch.zeros(1, device=self.device) # dim: (1,)
            full_asc = torch.cat([self.asc, zeros]) # dim: (J,)
            final_utility += full_asc[batch['alt'].long()] # dim: (B,)

        return final_utility

def _calculate_loss_long_deprecated(self, batch: dict[str, torch.Tensor]):
    """
    Helper function to calculate the negative log-likelihood loss.
    This version dynamically creates the mask for homogeneous choice problems.
    """
    # 1. Get Utilities from the forward pass
    utilities = self.forward(batch) # dim: (B,)

    # 2. Group Utilities by Choice Scenario
    chid_unique, chid_counts = torch.unique_consecutive(batch['chid'], return_counts=True) #both objects' dim: (B,)
    utilities_by_choice = torch.split(utilities, chid_counts.tolist()) #a length N tuple of tensors of J_n utilities
    padded_utilities = nn.utils.rnn.pad_sequence(utilities_by_choice, batch_first=True, padding_value=0) # dim: (B, J)

    # 3. DYNAMICALLY CREATE MASK (if needed)
    # Create a range tensor [0, 1, 2, ..., J_max-1]
    range_tensor = torch.arange(self.hparams.n_alternatives, device=self.device) # dim: (J,)

    # Use broadcasting: compare a (N, 1) tensor with a (J,) tensor
    # chid_counts is the number of real alternatives for each scenario.
    mask = range_tensor < chid_counts.unsqueeze(1) # dim: (B, J)

    # Where the mask is False (i.e., for padded, unreal alternatives),
    # set the utility to a large negative number.
    padded_utilities[~mask] = -1e9

    # 4. Calculate Log-Probabilities
    log_probs = F.log_softmax(padded_utilities, dim=1) # dim: (B, J)

    # 5. Get the Chosen Alternative for Each Scenario
    chosen_alt_indices = batch['alt'][batch['match'] == 1] # dim: (B,)
    if len(chid_unique) != len(chosen_alt_indices):
        raise ValueError(
            f"Data Integrity Error: The number of unique choice scenarios ({len(chid_unique)}) "
            f"does not match the number of chosen alternatives ({len(chosen_alt_indices)}) in this batch. "
            "Please check your dataset for choice situations where no alternative has 'match == 1'."
        )

    # 6. Calculate Negative Log-Likelihood Loss; not using F.cross_entropy due to the masking
    loss = F.nll_loss(log_probs, chosen_alt_indices)

    # --- Calculate Accuracy ---
    # Note: Accuracy is only meaningful if every real alternative is a possible choice.
    # For homogeneous choice with many alternatives, this might be less informative.
    preds = torch.argmax(log_probs, dim=1)
    acc = (preds == chosen_alt_indices).float().mean()

    return loss, acc

if __name__ == '__main__':
    pass
