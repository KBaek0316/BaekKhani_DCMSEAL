# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import optuna
from src.data_processing import DataHandler
from src.model import DCMSEALModel
from src.training import train_model

def objective(trial):
    # 1. Suggest hyperparameters for this trial from a defined search space.
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'embedding_dim': trial.suggest_int('embedding_dim', 4, 32),
        'hidden_layers': trial.suggest_int('hidden_layers', 1, 3),
        #... other hyperparameters for the neural network or training process
    }

    # 2. Instantiate the data handler and model with the suggested parameters.
    data_handler = DataHandler(config['data_paths'])
    model = DCMSEALModel(model_params=params)
    
    # 3. Load data and execute the training and evaluation loop.
    training_data = data_handler.load_and_prepare_data(split='train')
    validation_data = data_handler.load_and_prepare_data(split='test') # Or a dedicated validation set
    validation_metric = train_model(model, training_data, validation_data, training_params=params)

    # 4. Return the performance metric for Optuna to optimize.
    return validation_metric

# --- In the main part of the script ---
study = optuna.create_study(direction='maximize') # e.g., for accuracy
study.optimize(objective, n_trials=100)