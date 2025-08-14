# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import os
from pathlib import Path
import warnings
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
torch.set_float32_matmul_precision('medium') #Enable Tensor Cores for performance boost

# Import custom modules from the 'src' directory
from src.data_processing import load_and_preprocess_data
from src.data_loader import ChoiceDataset, choice_collate_fn
from src.dcm_seal_model import DCM_SEAL

def run_model(config:dict,data2use:str='Synthesized',verbose:bool=False):
    """Main function to run the DCM-SEAL model experiment."""

    # --- 1. Load and Preprocess Data, and Update Data-Driven Variable Options to Config ---
    print("--- Starting Data Preprocessing ---")
    data_dir = Path.cwd() / "data" / data2use
    train_df, test_df, train_x_emb, test_x_emb, discovered_embedding_dims = load_and_preprocess_data(
        config=config,data_dir=data_dir)

    # Add the discovered embedding dimensions to the main config.
    # This is crucial for initializing the model with the correct layer sizes.
    config["embedding_dims"] = discovered_embedding_dims

    # Update the segmentation network input dimension based on the processed data
    # This makes the config robust to the number of one-hot columns created
    seg_vars_one_hot = [col for col in train_df.columns if any(f"{s}_" in col for s in config["segmentation_vars_categorical"])]
    seg_vars_cont = config.get("segmentation_vars_continuous", [])
    config["segmentation_vars"] = seg_vars_one_hot + seg_vars_cont
    total_seg_vars = len(seg_vars_one_hot) + len(seg_vars_cont)
    hidden_dims = config.get("segmentation_hidden_dims", []) # e.g., [32, 16]
    config["segmentation_net_dims"] = [total_seg_vars] + hidden_dims + [config["n_latent_classes"]]
    print(f"Updated segmentation network dimensions to: {config['segmentation_net_dims']}")

    # --- 2. Create PyTorch Datasets and DataLoaders ---
    print("\n--- Creating Datasets and DataLoaders ---")
    train_dataset = ChoiceDataset(train_df, train_x_emb, config)
    test_dataset = ChoiceDataset(test_df, test_x_emb, config)

    if os.name=='posix':
        numwork=8 #recommended: 4*# of GPUs, diminishing return afterward
        persistent=True
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Grad strides do not match.*")
    else: #setting the above for Windows can actually slow down the process
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        numwork=0 # nonzero for Windows significantly slows down the process
        persistent=False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False, # can become unstable when True
        collate_fn=choice_collate_fn,
        num_workers = numwork,
        persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False, # No need to shuffle validation/test data at all
        collate_fn=choice_collate_fn,
        num_workers = numwork,
        persistent_workers=persistent
    )

    # --- 3. Initialize and Train the Model ---
    print("\n--- Initializing Model and Trainer ---")
    model = DCM_SEAL(config)

    # Configure a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',      # Monitor validation loss
        dirpath='checkpoints/',
        filename= data2use + '_{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,            # Save only the best model
        mode='min'               # Mode is 'min' because we want to minimize loss
    )

    # Initialize the Trainer for multi-GPU training
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=-1, # Use all available GPUs for training
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("logs/", name=data2use + "_experiment"),
        log_every_n_steps=10
    )

    print("\n--- Starting Model Training ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # --- FINAL TESTING LOGIC ---
    # This block will now only be executed by the main GPU process (rank 0),
    if trainer.global_rank == 0:
        print("\n--- Starting Model Testing on a Single Device ---")
        
        # Create a new trainer configured specifically for single-device testing.
        test_trainer = pl.Trainer(
            accelerator="auto", 
            devices=1,
            callbacks=[checkpoint_callback],
            logger=pl.loggers.TensorBoardLogger("logs/", name=data2use + "_experiment")
        )

        # Use ckpt_path="best" to automatically find and load the best model
        test_trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    best_score = checkpoint_callback.best_model_score.item()
    return best_score

#%% Run
if __name__ == "__main__":
    os.chdir('..')
    config = {
        # -- Data Processing Hyperparameters --
        "core_vars": ["x1","x2","x3"],
        "embedding_vars": ["e1","e2","o1"],
        "segmentation_vars_categorical": ["s1","s2"],
        "segmentation_vars_continuous": ["dist"],
        "test_size": 0.25,
        "random_state": 5723588,

        # -- Model Architecture Hyperparameters --
        "n_latent_classes": 3,
        "n_alternatives": 2,
        "choice_mode": "heterogeneous",
        "embedding_mode": "class-specific", 
        "segmentation_hidden_dims": [128, 256, 128],

        # -- Regularization and Optimizer Hyperparameters --
        "learning_rate": 0.002,
        "segmentation_dropout_rate": 0.2,
        "weight_decay_segmentation": 1e-2,
        "weight_decay_embedding": 1e-3,

        # -- Training Hyperparameters --
        "batch_size": 1024,
        "max_epochs": 10,
    }
    run_model(config)