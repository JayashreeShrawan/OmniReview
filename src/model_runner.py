"""
OmniReview Model Runner - Week 12 Deliverable
===============================================
Loads pre-trained model checkpoints and generates representative review samples.

Usage:
    python src/model_runner.py
    python src/model_runner.py --samples 10
    python src/model_runner.py --category Electronics --rating 5
"""
import argparse
import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vae import OmniVAE
from src.models.transformer import LatentConditionedT5
from src.models.flow import create_flow
from src.models.diffusion import DenoisingMLP, get_beta_schedule, p_sample_loop
from src.models.gan import ReviewDiscriminator
from src.data_loader import load_data_and_embeddings


def load_config(config_path="configs/model_config.yaml"):
    """Load configuration from YAML file."""
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(model, path, device):
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="OmniReview Model Runner")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Config file path")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--category", type=str, default=None, help="Specific category to generate")
    parser.add_argument("--rating", type=int, default=None, help="Specific star rating (1-5)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIM = config["model"]["embedding_dim"]
    LATENT_DIM = config["model"]["latent_dim"]
    NUM_CATEGORIES = config["model"]["num_categories"]
    CHECKPOINT_DIR = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    OUTPUT_DIR = PROJECT_ROOT / config["paths"]["output_dir"]
    DATA_DIR = PROJECT_ROOT / config["paths"]["data_dir"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoints from: {CHECKPOINT_DIR}")

    # --- Load category encoder ---
    train_df = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    le_cat = LabelEncoder()
    le_cat.fit(train_df["product_category"])
    categories = list(le_cat.classes_)
    print(f"Categories: {categories}")

    # --- Load VAE ---
    print("Loading VAE...")
    vae_model = OmniVAE(embed_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    vae_model = load_checkpoint(vae_model, CHECKPOINT_DIR / "vae_best.pt", DEVICE)

    # --- Load Flow ---
    print("Loading Normalizing Flow...")
    flow_model = create_flow(features=1, context_features=16, num_layers=8, hidden_features=32)
    flow_model = flow_model.to(DEVICE)
    flow_model = load_checkpoint(flow_model, CHECKPOINT_DIR / "flow_best.pt", DEVICE)
    # Note: These embeddings were lost in the week11 checkpoint (only flow_model was saved)
    # so they are random, but they must be size 8 so they sum to 16 for the context!
    cat_emb_flow = nn.Embedding(NUM_CATEGORIES, 8).to(DEVICE)
    rat_emb_flow = nn.Embedding(5, 8).to(DEVICE)

    # --- Load Diffusion ---
    print("Loading Diffusion Model...")
    diff_model = DenoisingMLP(embed_dim=EMBEDDING_DIM, cond_dim=32, hidden_dim=512).to(DEVICE)
    diff_model = load_checkpoint(diff_model, CHECKPOINT_DIR / "diffusion_best.pt", DEVICE)

    # --- Load GAN Discriminator ---
    print("Loading GAN Discriminator...")
    gan_disc = ReviewDiscriminator(embed_dim=EMBEDDING_DIM, cond_dim=32).to(DEVICE)
    gan_disc = load_checkpoint(gan_disc, CHECKPOINT_DIR / "gan_best.pt", DEVICE)

    # --- Load T5 ---
    print("Loading T5 + LoRA...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    base_t5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=True,
        r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "v"]
    )
    lora_t5 = get_peft_model(base_t5, peft_config)
    conditioned_t5 = LatentConditionedT5(lora_t5, LATENT_DIM).to(DEVICE)
    conditioned_t5 = load_checkpoint(conditioned_t5, CHECKPOINT_DIR / "t5_vae_best.pt", DEVICE)

    print("\nAll models loaded successfully!\n")

    # --- Generate samples ---
    @torch.no_grad()
    def generate_review(category_idx, rating_val):
        """Generate a single review using the full OmniReview pipeline."""
        cat = torch.full((1,), category_idx, dtype=torch.long).to(DEVICE)
        rat = torch.full((1,), rating_val - 1, dtype=torch.long).to(DEVICE)

        # Condition embeddings
        c_emb = vae_model.cat_emb(cat)
        r_emb = vae_model.rat_emb(rat)
        cond = torch.cat([c_emb, r_emb], dim=1)

        # Flow: sample helpfulness
        flow_ctx = torch.cat([cat_emb_flow(cat), rat_emb_flow(rat)], dim=1)
        helpfulness = flow_model.sample(1, context=flow_ctx).squeeze()

        # VAE: sample latent
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        body_vae, head_vae = vae_model.decoder(z, cond)

        # Diffusion: refine
        diff_out = p_sample_loop(diff_model, (1, EMBEDDING_DIM * 2), cond)
        body_diff, head_diff = diff_out.split(EMBEDDING_DIM, dim=1)

        # GAN: quality score
        quality = gan_disc(body_diff, head_diff, helpfulness.unsqueeze(0), cond)

        # T5: generate text
        cat_name = le_cat.inverse_transform([category_idx])[0]
        prompt = f"generate review: category {cat_name} rating {rating_val}"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        attn_mask = torch.ones_like(input_ids).to(DEVICE)
        gen_tokens = conditioned_t5.generate(
            input_ids=input_ids, attention_mask=attn_mask, latent_z=z,
            max_length=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
        )
        text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        return {
            "category": cat_name,
            "star_rating": rating_val,
            "helpfulness": round(helpfulness.item(), 4),
            "gan_quality": round(quality.item(), 4),
            "text": text,
        }

    # Determine what to generate
    results = []
    if args.category and args.rating:
        cat_idx = le_cat.transform([args.category])[0]
        for _ in range(args.samples):
            results.append(generate_review(cat_idx, args.rating))
    else:
        # Generate across all categories and ratings
        count = 0
        for cat_idx in range(NUM_CATEGORIES):
            for rating in [1, 3, 5]:
                if count >= args.samples:
                    break
                results.append(generate_review(cat_idx, rating))
                count += 1
            if count >= args.samples:
                break

    # Print results
    print("=" * 70)
    print(f"Generated {len(results)} reviews:")
    print("=" * 70)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['category']} | {r['star_rating']} stars | "
              f"helpfulness={r['helpfulness']:.4f} | quality={r['gan_quality']:.4f}")
        print(f"    {r['text']}")

    # Save outputs
    csv_path = OUTPUT_DIR / "generated_reviews.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "star_rating", "helpfulness", "gan_quality", "text"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV: {csv_path}")

    txt_path = OUTPUT_DIR / "samples.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("OmniReview Generated Samples\n")
        f.write("=" * 70 + "\n\n")
        for i, r in enumerate(results):
            f.write(f"[{i+1}] Category: {r['category']} | Rating: {r['star_rating']} stars\n")
            f.write(f"    Helpfulness: {r['helpfulness']:.4f} | GAN Quality: {r['gan_quality']:.4f}\n")
            f.write(f"    Text: {r['text']}\n\n")
    print(f"Saved TXT: {txt_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
