"""Data loading utilities for OmniReview pipeline."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class EmbeddedReviewDataset(Dataset):
    """PyTorch Dataset with precomputed S-BERT embeddings."""
    def __init__(self, df, body_emb, head_emb):
        self.df = df.reset_index(drop=True)
        self.body_emb = torch.tensor(body_emb, dtype=torch.float32)
        self.head_emb = torch.tensor(head_emb, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "body_emb": self.body_emb[idx],
            "head_emb": self.head_emb[idx],
            "category": torch.tensor(row["category_encoded"], dtype=torch.long),
            "rating": torch.tensor(row["star_rating"] - 1, dtype=torch.long),
            "helpfulness": torch.tensor(row["helpfulness_scaled"], dtype=torch.float32).unsqueeze(0),
        }


def load_data_and_embeddings(data_dir="data", batch_size=64):
    """Load preprocessed parquet splits and S-BERT embeddings.

    Returns:
        train_df, val_df, test_df, train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    train_df = pd.read_parquet(data_dir / "train_clean.parquet")
    val_df = pd.read_parquet(data_dir / "val_clean.parquet")
    test_df = pd.read_parquet(data_dir / "test_clean.parquet")

    train_body = np.load(data_dir / "train_body_emb.npy")
    train_head = np.load(data_dir / "train_head_emb.npy")
    val_body = np.load(data_dir / "val_body_emb.npy")
    val_head = np.load(data_dir / "val_head_emb.npy")
    test_body = np.load(data_dir / "test_body_emb.npy")
    test_head = np.load(data_dir / "test_head_emb.npy")

    train_ds = EmbeddedReviewDataset(train_df, train_body, train_head)
    val_ds = EmbeddedReviewDataset(val_df, val_body, val_head)
    test_ds = EmbeddedReviewDataset(test_df, test_body, test_head)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader
