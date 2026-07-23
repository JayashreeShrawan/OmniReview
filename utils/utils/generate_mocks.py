"""
Script to generate mock checkpoints and mock data for testing the OmniReview pipeline locally.
Run this script to initialize dummy weights and data, avoiding the need to download large datasets or train models for hours.
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from transformers import T5ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vae import OmniVAE
from src.models.transformer import LatentConditionedT5
from src.models.flow import create_flow
from src.models.diffusion import DenoisingMLP
from src.models.gan import ReviewDiscriminator
import yaml

def load_config(config_path="configs/model_config.yaml"):
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)

def save_mock_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    print(f"Saved mock checkpoint: {path}")

def main():
    print("Generating mock checkpoints and data...")
    config = load_config()
    
    EMBEDDING_DIM = config["model"]["embedding_dim"]
    LATENT_DIM = config["model"]["latent_dim"]
    NUM_CATEGORIES = config["model"]["num_categories"]
    CHECKPOINT_DIR = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    DATA_DIR = PROJECT_ROOT / config["paths"]["data_dir"]

    # 1. Generate Mock Data (Parquet)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    categories = ["Electronics", "Books", "Camera", "Apparel", "Digital_Video_Games"]
    mock_df = pd.DataFrame({"product_category": categories * 2})
    mock_df.to_parquet(DATA_DIR / "train_clean.parquet")
    print(f"Saved mock data: {DATA_DIR / 'train_clean.parquet'}")

    # 2. Generate Mock Checkpoints
    device = torch.device("cpu")
    
    # VAE
    vae = OmniVAE(embed_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM).to(device)
    save_mock_checkpoint(vae, CHECKPOINT_DIR / "vae_best.pt")

    # Flow
    flow = create_flow(features=1, context_features=32, num_layers=8, hidden_features=64).to(device)
    save_mock_checkpoint(flow, CHECKPOINT_DIR / "flow_best.pt")

    # Diffusion
    diff = DenoisingMLP(input_dim=EMBEDDING_DIM * 2, cond_dim=32, hidden_dim=512).to(device)
    save_mock_checkpoint(diff, CHECKPOINT_DIR / "diffusion_best.pt")

    # GAN
    gan = ReviewDiscriminator(embed_dim=EMBEDDING_DIM, cond_dim=32).to(device)
    save_mock_checkpoint(gan, CHECKPOINT_DIR / "gan_best.pt")

    # T5 + LoRA
    print("Initializing mock T5 (this will download t5-small if not cached)...")
    base_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "v"]
    )
    lora_t5 = get_peft_model(base_t5, peft_config)
    conditioned_t5 = LatentConditionedT5(lora_t5, LATENT_DIM).to(device)
    save_mock_checkpoint(conditioned_t5, CHECKPOINT_DIR / "t5_vae_best.pt")

    print("\nSuccessfully generated all mocks! You can now run `python src/model_runner.py`.")

if __name__ == "__main__":
    main()
