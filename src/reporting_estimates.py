# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""
import torch
import pandas as pd

def extract_betas(model, config):
    """
    Returns a DataFrame of betas with columns ordered:
      config['core_vars'] + config['embedding_vars'].
    Uses model.beta (K x D or D).
    """
    core_names = list(config.get("core_vars", []))
    emb_names  = list(config.get("embedding_vars", []))
    names = core_names + emb_names

    beta = model.beta  # provided by the @property above
    if isinstance(beta, torch.nn.Parameter):
        beta = beta.data
    if beta.ndim == 1:
        beta = beta.unsqueeze(0)  # (D,) -> (1, D)

    K, D = beta.shape
    if D != len(names):
        raise ValueError(f"Beta width {D} != expected {len(names)} "
                         "(len(core_vars)+len(embedding_vars)).")

    df = pd.DataFrame(beta.detach().cpu().numpy(), columns=names)
    df.index = ["beta"] if K == 1 else [f"class_{i+1}" for i in range(K)]
    return df