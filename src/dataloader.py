"""
data_loader.py

Loads the processed dataset used by the OmniReview pipeline.
"""

import pandas as pd


def load_dataset(filepath):
    """
    Load the processed dataset.

    Args:
        filepath (str): Path to processed CSV.

    Returns:
        pandas.DataFrame
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data)} samples.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None