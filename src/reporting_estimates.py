# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""
import torch
import numpy as np
import pandas as pd
from collections.abc import Mapping
from typing import List, Dict, Optional

def extract_betas(model, config):
    """
    Extracts the betas (parameters) from the trained model for reporting purposes.
    Returns the **constrained** betas used in the forward pass, not the raw, unconstrained betas.
    """
    
    betas = model.beta  # Get constrained betas (model.beta already applies softplus and non-positivity constraints)
    beta_list = []
    
    core_names = list(config.get("core_vars", []))
    emb_names  = list(config.get("embedding_vars", []))
    names = core_names + emb_names
    
    # --- Extract Core Betas ---
    if len(config["core_vars"]) > 0:
        core_betas = betas[:, :len(config["core_vars"])]  # (K, C)
        for i, var in enumerate(config["core_vars"]):
            core_betas_df = core_betas[:, i].cpu().detach().numpy()
            beta_list.append(core_betas_df)

    # --- Extract Embedding Betas ---
    if len(config["embedding_vars"]) > 0:
        emb_betas = betas[:, len(config["core_vars"]):]  # (K, E)
        for i, var in enumerate(config["embedding_vars"]):
            emb_betas_df = emb_betas[:, i].cpu().detach().numpy()
            beta_list.append(emb_betas_df)

    beta_df = pd.DataFrame(np.column_stack(beta_list), columns=names)

    # Optionally include the raw betas as well (for debugging purposes)
    beta_raw = model.beta_raw  # This gives unconstrained betas if needed

    # Return the dictionary containing the betas as numpy arrays
    return beta_df, beta_raw

def _to_tensor(x) -> torch.Tensor:
    w = x.weight if hasattr(x, "weight") else x
    if isinstance(w, torch.nn.Parameter):
        w = w.data
    return w.detach().cpu()


def _row_labels_from_hparams(hparams: dict, key: str = "embedding_labels") -> List[str]:
    """
    Flatten the (count, labels) blocks from hparams[key] (an OrderedDict):
      { var: (count, [labels...]), ... }  -->  [labels...] in block order
    Assumes labels are already human-readable and in training order.
    """
    emb = hparams.get(key, None)
    if emb is None:
        raise KeyError(f"hparams['{key}'] not found.")
    rows: List[str] = []
    for var, (cnt, labels) in emb.items():
        if not isinstance(labels, (list, tuple)) or len(labels) != cnt:
            # fallback if something is off in the entry
            labels = [f"{var}[{i}]" for i in range(cnt)]
        rows.extend(map(str, labels))
    return rows


def extract_embedding(
    model,
    alt_names: Optional[List[str]] = None,
    hparams_key: str = "embedding_labels",
) -> Dict[str, pd.DataFrame]:
    """
    Return {'global': df} for shared embeddings, or {'class_1': df, ..., 'class_K': df}
    for class-specific embeddings. Each df mirrors the embedding matrix shape (n_levels, n_alts).
    Row labels come directly from model.hparams[hparams_key] (always available per your setup).
    """
    # 1) locate embedding container
    emb = getattr(model, "embedding_layers", None)
    if emb is None and isinstance(model, Mapping):
        emb = model.get("embedding_layers", None)
    if emb is None:
        raise AttributeError("embedding_layers not found on model (attr or mapping key).")

    # 2) build row labels from hparams (single source of truth)
    row_labels = _row_labels_from_hparams(model.hparams, key=hparams_key)

    def make_df(W: torch.Tensor, tag: str) -> pd.DataFrame:
        if W.ndim != 2:
            raise ValueError(f"{tag}: expected 2D (n_levels, n_alts), got {tuple(W.shape)}")
        n_levels, n_alts = W.shape

        rows = row_labels if len(row_labels) == n_levels else [f"level_{i}" for i in range(n_levels)]
        cols = alt_names if (alt_names and len(alt_names) == n_alts) else [f"alt_{j+1}" for j in range(n_alts)]

        return pd.DataFrame(W.numpy(), index=rows, columns=cols)

    # 3) emit one DataFrame per class, or one 'global'
    tables: Dict[str, pd.DataFrame] = {}
    if isinstance(emb, torch.nn.ModuleList):
        for k, item in enumerate(emb):
            W = _to_tensor(item)
            tables[f"class_{k+1}"] = make_df(W, f"class_{k+1}")
    else:
        W = _to_tensor(emb)
        tables["global"] = make_df(W, "global")
    return tables


if __name__ == '__main__':
    pass
'''
    trained_model.hparams['segmentation_vars']
    trained_model.hparams['embedding_vars']
    trained_model.hparams['embedding_dims']
    trained_model.hparams['embedding_labels']
'''