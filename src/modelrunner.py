"""
model_runner.py

Runs the complete OmniReview inference pipeline.
"""

from data_loader import load_dataset

# Import your pipeline
from models.pipeline import generate_omnireview

OUTPUT_FILE = "outputs/generated_reviews.txt"

DATA_PATH = "data/processed/processed_reviews.csv"


def save_outputs(results):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for i, review in enumerate(results):

            f.write(f"Sample {i+1}\n")
            f.write(f"Category : {review['category']}\n")
            f.write(f"Rating : {review['star_rating']}\n")
            f.write(f"Helpfulness : {review['helpfulness']:.3f}\n")
            f.write(f"GAN Quality : {review['gan_quality']:.3f}\n")
            f.write(f"Review :\n{review['text']}\n")
            f.write("-"*60 + "\n")

    print("Outputs saved successfully.")


def main():

    print("Loading dataset...")

    dataset = load_dataset(DATA_PATH)

    if dataset is None:

        return

    print("Running OmniReview pipeline...")

    results = generate_omnireview(
        category_idx=0,
        rating_val=5,
        num_samples=10
    )

    save_outputs(results)

    print("Pipeline completed successfully.")


if __name__ == "__main__":

    main()