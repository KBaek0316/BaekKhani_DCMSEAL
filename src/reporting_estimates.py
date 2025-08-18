# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""
import torch
import pandas as pd
from collections.abc import Mapping
from typing import List, Dict, Optional

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