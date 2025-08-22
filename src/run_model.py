# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import os
from pathlib import Path

if __name__ == "__main__":
    if Path.cwd().name=='src':
        os.chdir("..")

import warnings
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
torch.set_float32_matmul_precision('medium')  # Enable Tensor Cores for performance boost
torch.backends.cudnn.benchmark = True         # Let cuDNN pick fastest kernels for fixed shapes


# Import custom modules from the 'src' directory
from src.data_processing import load_and_preprocess_data
from src.data_loader import PaddedChoiceDataset
from src.dcm_seal_model import DCM_SEAL

def run_model(config:dict,data2use:str='Synthesized',verbose:bool=False):
    """Main function to run the DCM-SEAL model experiment."""

    # --- 1. Load and Preprocess Data, and Update Data-Driven Variable Options to Config ---
    print("--- Starting Data Preprocessing ---")
    data_dir = Path.cwd() / "data" / data2use
    train_df, test_df, train_x_emb, test_x_emb, discovered_embedding_dict = load_and_preprocess_data(
        config=config,data_dir=data_dir)

    # Add the discovered embedding dimensions to the main config.
    # This is crucial for initializing the model with the correct layer sizes.
    config["embedding_dims"] = discovered_embedding_dict['dims']
    config["embedding_labels"] = discovered_embedding_dict['labels']

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
    train_dataset = PaddedChoiceDataset(train_df, train_x_emb, config)
    test_dataset = PaddedChoiceDataset(test_df, test_x_emb, config)

    if os.name=='nt':
        numwork=0 # nonzero for Windows significantly slows down the process
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

    else:
        numwork=4 #recommended: 4*# of GPUs, diminishing return afterward; restricted GPU usage to 1
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Grad strides do not match.*")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=numwork,
        pin_memory=True
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=numwork,
        pin_memory=True
        )

    # --- 3. Initialize and Train the Model ---
    print("\n--- Initializing Model and Trainer ---")
    model = DCM_SEAL(config)
    #store identified labels into model
    

    # Configure a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',      # Monitor validation loss
        dirpath='checkpoints/',
        filename= data2use + '_{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,            # Save only the best model
        mode='min'               # Mode is 'min' because we want to minimize loss
    )
    
    early_stop = EarlyStopping( #for Optuna speedup
        monitor="val_loss",    # or a metric you prefer
        mode="min",
        patience=3,            # tune this if your curves are noisy
        min_delta=0.0,
        check_finite=True,
    )
    
    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if use_gpu else "cpu", #"auto",
        devices=1, #removed multi-GPU DDP (slower)
        precision="16-mixed" if use_gpu else 32,
        callbacks=[checkpoint_callback, early_stop],
        logger=pl.loggers.TensorBoardLogger("logs/", name=data2use + "_experiment"),
        log_every_n_steps=10,
        enable_model_summary=False
)


    print("\n--- Starting Model Training ---")
    if verbose:
        batch = next(iter(train_loader))
        print(
            batch['core_features'].shape,  # (B, J_max, C)
            batch['mask'].shape,           # (B, J_max)
            batch['seg_features'].shape,   # (B, S)
            batch['x_emb'].shape,          # (B, E)
            batch['choice'].shape          # (B,)
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    best_score = checkpoint_callback.best_model_score.item()
    return model, best_score

#%% Smoke Run
if __name__ == "__main__":
    data2use='TwinCitiesPath'
    match data2use:
        case 'TwinCitiesPath':
            config = {
                # -- Data Processing Hyperparameters --
                "core_vars": ["tway","iv", "wt", "wk","nTrans","PS"],
                "non_positive_core_vars":['iv','nTrans'],
                "embedding_vars": ["summer","dayofweek","plan","realtime","access","egress","oppo","hr"],
                "segmentation_vars_categorical": ["hhsize","HHcomp","white","visitor","worktype","stu","engflu","age","income","disability","gender","choicerider","purpose"],
                "segmentation_vars_continuous": [],
                "test_size": 0.2,
                "random_state": 5723588,

                # -- Model Architecture Hyperparameters --
                "n_latent_classes": 2,
                "n_alternatives": 5,
                "choice_mode": "homogeneous",
                "embedding_mode": "class-specific", 
                "segmentation_hidden_dims": [128, 256, 128],

                # -- Regularization and Optimizer Hyperparameters --
                "learning_rate": 0.002,
                "segmentation_dropout_rate": 0.2,
                "weight_decay_segmentation": 1e-2,
                "weight_decay_embedding": 1e-3,

                # -- Training Hyperparameters --
                "batch_size": 2048,
                "max_epochs": 50,
            }
        case 'Synthesized':
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
                "embedding_mode": "shared", 
                "segmentation_hidden_dims": [128, 256, 128],

                # -- Regularization and Optimizer Hyperparameters --
                "learning_rate": 0.002,
                "segmentation_dropout_rate": 0.2,
                "weight_decay_segmentation": 1e-2,
                "weight_decay_embedding": 1e-3,

                # -- Training Hyperparameters --
                "batch_size": 512,
                "max_epochs": 50,
            }

    trained_model, bestobj = run_model(config,data2use,True)
    from src.reporting_estimates import extract_betas, extract_embedding
    beta, beta_raw = extract_betas(trained_model, config)
    emb_df = extract_embedding(trained_model,config)
    print(beta)
    print(beta_raw)
    print(emb_df)
