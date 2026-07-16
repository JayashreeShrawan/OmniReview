"""
helpers.py

Utility functions used across the OmniReview project.
"""

import os
import random
import numpy as np
import torch


def create_output_directory(path="outputs"):
    """
    Create output directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def save_generated_reviews(results, output_file):
    """
    Save generated reviews to a text file.
    """

    with open(output_file, "w", encoding="utf-8") as f:

        for i, review in enumerate(results, start=1):

            f.write(f"Sample {i}\n")
            f.write(f"Category      : {review['category']}\n")
            f.write(f"Star Rating   : {review['star_rating']}\n")
            f.write(f"Helpfulness   : {review['helpfulness']:.3f}\n")
            f.write(f"GAN Quality   : {review['gan_quality']:.3f}\n")
            f.write("Generated Review:\n")
            f.write(review["text"] + "\n")
            f.write("-" * 70 + "\n")


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_status(message):
    """
    Print formatted pipeline status messages.
    """
    print(f"[OmniReview] {message}")
    