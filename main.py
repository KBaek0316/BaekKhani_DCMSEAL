# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""
#%% initial settings
# Comment/uncomment the following based on your debugging needs
DATA2USE="Synthesized" #['TwinCitiesPath', 'SwissMetro', 'Synthesized']
# see results with anaconda: tensorboard --logdir 'LogSavedDirectory'/

import os
from pathlib import Path
import math
import warnings

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Grad strides do not match.*")

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

if globals().get('DATA2USE') is None:
    DATASETS=[p.name for p in (WPATH/'data').iterdir() if p.is_dir()]
    DATA2USE=input(f"Type a dataset you would like to use from {DATASETS} (case-sensitive): ")


#%% main
import optuna
import torch
torch.set_float32_matmul_precision('medium')
from src.run_model import run_model
import pandas as pd


def objective(trial: optuna.trial.Trial) -> float:
    """
    The Optuna objective function with dynamic epoch and batch size calculation.
    """
    match DATA2USE:
        case 'TwinCitiesPath':
            train_size= 56225 #needed to calculate batch size in Optuna; long format's length is ok
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
                "choice_mode": "homogeneous", #'heterogeneous' or 'homogeneous'.
            }
        case 'Synthesized':
            train_size= 25000
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
            }
    # --- Define the Hyperparameter Search Space ---

    # 1. Determining epochs and batch size
    total_updates = trial.suggest_categorical("total_updates", [300, 600, 900, 1200, 1500])
    updates_per_epoch = trial.suggest_categorical("updates_per_epoch", [1, 10, 20, 40]) # 1: full-batch
    
    # Deterministically calculate epochs and batch size
    max_epochs = min(max(50,total_updates // updates_per_epoch),500)
    batch_size = max(128, math.ceil(train_size / updates_per_epoch))
    
    # 2. Optimizer and Regularization Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
    weight_decay_segmentation = trial.suggest_float("weight_decay_segmentation", 1e-4, 1e-1, log=True)
    weight_decay_embedding = trial.suggest_float("weight_decay_embedding", 1e-5, 1e-2, log=True)
    segmentation_dropout_rate = trial.suggest_float("segmentation_dropout_rate", 0.0, 0.6)

    # 3. Architectural Hyperparameters
    embedding_mode = trial.suggest_categorical("embedding_mode", ["shared", "class-specific"]) #maybe not optuna?
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 4)
    hidden_dims = [trial.suggest_categorical(f"n_nodes_layer_{i}", [64, 128, 256]) for i in range(num_hidden_layers)]

    # --- Update the global config with dynamically suggested hyperparameters ---
    config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "weight_decay_segmentation": weight_decay_segmentation,
        "weight_decay_embedding": weight_decay_embedding,
        "segmentation_dropout_rate": segmentation_dropout_rate,
        "embedding_mode": embedding_mode,
        "segmentation_hidden_dims": hidden_dims,
    })

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  - Derived params: epochs={max_epochs}, batch_size={batch_size}")

    # Optuna will try to MINIMIZE the value returned by this function.
    _, val_loss = run_model(config, DATA2USE)
    
    return val_loss

#%% Run
if __name__ == "__main__":
    # --- Set up Project Directory ---
    STORAGE_NAME='sqlite:///optunaStudies.db'
    
    # --- Create and Run the Optuna Study ---
    study = optuna.create_study(
        direction="minimize",
        study_name="dcm_seal_hyperparameter_search",
        storage=STORAGE_NAME,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=200)
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    '''
    STORAGE_NAME='sqlite:///smokeStudies.db'
    finalStudy=optuna.create_study(direction="minimize",study_name="dcm_seal_hyperparameter_search",storage=STORAGE_NAME,load_if_exists=True)
    finalResults = finalStudy.trials_dataframe()
    finalResults['duration']=finalResults['duration'].dt.total_seconds().astype(int)
    finalResults = finalResults.sort_values("value").reset_index(drop=True)
    '''