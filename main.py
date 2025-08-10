# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import os
from pathlib import Path
if '__file__' in globals():
    WPATH = Path(__file__).resolve().parent
elif os.name=='posix': #linux server; Windows='nt'
    WPATH='/export/scratch/users/baek0040/git/TCML/MLTraining'
else:
    if os.environ['USERPROFILE']==r'C:\Users\baek0040':
        WPATH=Path('C:/Users/baek0040/Documents/GitHub/BaekKhani_DCMSEAL').resolve()
    else:
        WPATH=Path('C:/git/BaekKhani_DCMSEAL').resolve()
os.chdir(WPATH)
data2use="empirical" #'empirical' or 'synthesized'
WPATH / "data" / data2use

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import custom modules from the 'src' directory
from src.data_processing import load_and_preprocess_data
from src.dataloader import ChoiceDataset
from src.dcm_seal_model import DCM_SEAL

def main():
    """Main function to run the DCM-SEAL model experiment."""
    # --- 1. Master Configuration ---
    # This dictionary controls everything about the experiment.
    # It will be passed to the data processor and the model.
    if data2use=="empirical":
        config = { #segmentation_net_dims will be determined later
            # -- Data Processing Hyperparameters --
            "embedding_vars": ["purpose", "hr", "worktype"],
            "segmentation_vars_categorical": ["age", "income", "gender"],
            "segmentation_vars_continuous": [], # Add continuous seg vars here if any
            "core_vars": ["cost", "iv", "ov", "nTrans"],
            "test_size": 0.2,
            "random_state": 42,

            # -- Model Architecture Hyperparameters --
            "n_latent_classes": 4,
            "n_alternatives": 3,
            "choice_mode": "heterogeneous",           # 'heterogeneous' or 'homogeneous'
            "embedding_mode": "class-specific",       # 'shared' or 'class-specific'
            "segmentation_hidden_dims": [128, 256, 128], # hidden layers

            # -- Regularization and Optimizer Hyperparameters --
            "learning_rate": 0.001,
            "segmentation_dropout_rate": 0.5,
            "weight_decay_segmentation": 0.01,
            "weight_decay_embedding": 0.0001,

            # -- Training Hyperparameters --
            "batch_size": 256,
            "max_epochs": 100,
        }
    else:
        config = {
        }

    # --- 2. Load and Preprocess Data ---
    print("--- Starting Data Preprocessing ---")
    data_dir = WPATH / "data" / data2use
    train_df, test_df, train_x_emb, test_x_emb, discovered_embedding_dims = load_and_preprocess_data(
        config=config,data_dir=data_dir)

    # Add the discovered embedding dimensions to the main config.
    # This is crucial for initializing the model with the correct layer sizes.
    config["embedding_dims"] = discovered_embedding_dims
    
    # Update the segmentation network input dimension based on the processed data
    # This makes the config robust to the number of one-hot columns created
    seg_vars_one_hot = [col for col in train_df.columns if any(f"{s}_" in col for s in config["segmentation_vars_categorical"])]
    seg_vars_cont = config.get("segmentation_vars_continuous", [])
    total_seg_vars = len(seg_vars_one_hot) + len(seg_vars_cont)
    hidden_dims = config.get("segmentation_hidden_dims", []) # e.g., [32, 16]
    config["segmentation_net_dims"] = [total_seg_vars] + hidden_dims + config["n_latent_classes"]
    print(f"Updated segmentation network dimensions to: {config['segmentation_net_dims']}")

    # --- 3. Create PyTorch Datasets and DataLoaders ---
    print("\n--- Creating Datasets and DataLoaders ---")
    train_dataset = ChoiceDataset(train_df, train_x_emb, config)
    test_dataset = ChoiceDataset(test_df, test_x_emb, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffle training data for better learning
        num_workers = max(1, os.cpu_count() - 2)  # Use multiple CPU cores
    )
    # No need to shuffle validation/test data
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers = max(1, os.cpu_count() - 2)
    )

    # --- 4. Initialize and Train the Model ---
    print("\n--- Initializing Model and Trainer ---")
    model = DCM_SEAL(config)

    # Configure a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',      # Monitor validation loss
        dirpath='checkpoints/',
        filename='dcm-seal-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,            # Save only the best model
        mode='min'               # Mode is 'min' because we want to minimize loss
    )

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",  # Automatically uses GPU if available
        devices=-1, # Use all available GPUs if there are
        callbacks=[checkpoint_callback], # Add the checkpoint callback
        logger=pl.loggers.TensorBoardLogger("logs/", name="dcm_seal_experiment")
    )

    print("\n--- Starting Model Training ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    print("\n--- Starting Model Testing ---")
    # The trainer.test() method will automatically load the best model checkpoint.
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    main()