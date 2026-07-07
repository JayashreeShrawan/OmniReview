"""
Data preprocessing pipeline for OmniReview.

Tasks:
1. Load Amazon Reviews dataset
2. Filter verified purchases
3. Remove missing values
4. Compute helpfulness ratio
5. Export cleaned dataset
"""

import pandas as pd

def load_data(path):
    return pd.read_csv(path, sep="\t")

if __name__ == "__main__":
    print("OmniReview preprocessing pipeline")
