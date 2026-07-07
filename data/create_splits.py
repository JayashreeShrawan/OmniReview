"""
create_splits.py

Creates train, validation, and test splits for the
OmniReview dataset.

Dataset:
Amazon US Customer Reviews Dataset

Split Ratio:
- Training: 80%
- Validation: 10%
- Test: 10%
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def create_data_splits(df):
    """
    Split the dataset into training, validation,
    and testing datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset after preprocessing.

    Returns
    -------
    train_df : pandas.DataFrame
    val_df : pandas.DataFrame
    test_df : pandas.DataFrame
    """

    # First split: 80% training, 20% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    # Second split: temporary -> validation and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    print("Creating train, validation, and test datasets...")
