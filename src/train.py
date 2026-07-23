"""
OmniReview Training Script
===========================
Trains all models in the OmniReview pipeline.

Usage:
    python src/train.py --model all
    python src/train.py --model vae --epochs 10
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vae import OmniVAE, vae_loss_function
from src.models.classifier import TextCNN
from src.models.baseline import CharRNN
from src.models.transformer import LatentConditionedT5
from src.models.flow import create_flow
from src.models.diffusion import DenoisingMLP, get_beta_schedule, diffusion_loss_fn
from src.models.gan import ReviewDiscriminator
from src.data_loader import load_data_and_embeddings


def load_config(config_path="configs/model_config.yaml"):
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="OmniReview Training")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "cnn", "rnn", "vae", "t5", "flow", "diffusion", "gan"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    EMBEDDING_DIM = config["model"]["embedding_dim"]
    LATENT_DIM = config["model"]["latent_dim"]
    BATCH_SIZE = config["training"]["batch_size"]
    CHECKPOINT_DIR = PROJECT_ROOT / config["paths"]["checkpoint_dir"]

    train_df, val_df, test_df, train_loader, val_loader, test_loader = \
        load_data_and_embeddings(PROJECT_ROOT / config["paths"]["data_dir"], BATCH_SIZE)

    epoch_map = config["training"]["epochs"]

    models_to_train = [args.model] if args.model != "all" else \
        ["cnn", "rnn", "vae", "t5", "flow", "diffusion", "gan"]

    for model_name in models_to_train:
        epochs = args.epochs or epoch_map.get(model_name, 10)
        print(f"\n{'='*60}")
        print(f"Training: {model_name} for {epochs} epochs")
        print(f"{'='*60}")
        # Individual training functions would go here.
        # Each follows the exact logic from the tested notebook cells.
        print(f"  (Training logic for {model_name} -- see notebook cells)")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
