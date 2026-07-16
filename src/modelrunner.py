"""
model_runner.py

Runs the complete OmniReview inference pipeline.

This script:
1. Loads the processed dataset
2. Loads the OmniReview generation pipeline
3. Generates sample product reviews
4. Saves the generated outputs
"""

from data_loader import load_dataset
from models.pipeline import generate_omnireview
from utils.helpers import (
    create_output_directory,
    save_generated_reviews,
    set_seed,
    print_status,
)

OUTPUT_FILE = "outputs/generated_reviews.txt"
DATA_PATH = "data/processed/processed_reviews.csv"


def main():
    """Run the complete OmniReview pipeline."""

    # Set random seed
    set_seed(42)

    # Create outputs directory
    create_output_directory()

    # Load dataset
    print_status("Loading processed dataset...")
    dataset = load_dataset(DATA_PATH)

    if dataset is None:
        print("Dataset could not be loaded.")
        return

    # Run generation
    print_status("Running OmniReview pipeline...")

    results = generate_omnireview(
        category_idx=0,
        rating_val=5,
        num_samples=10
    )

    # Save generated reviews
    save_generated_reviews(results, OUTPUT_FILE)

    print_status("Pipeline completed successfully.")
    print_status(f"Generated reviews saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()