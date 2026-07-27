"""Shared utility functions for OmniReview."""
import time
import torch
import numpy as np


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"  [{self.label}] {elapsed:.2f}s")


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save a model checkpoint to disk."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load a model checkpoint from disk."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("loss", 0)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
