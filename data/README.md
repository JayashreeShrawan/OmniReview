# Data

This folder contains the dataset preprocessing pipeline and supporting scripts for the OmniReview project.

## Dataset

Amazon US Customer Reviews Dataset

Source:
https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

## Data Processing Steps

- Download dataset
- Filter verified purchases
- Remove incomplete records
- Compute helpfulness ratio
- Stratified sampling
- Generate Sentence-BERT embeddings
- Split into training, validation, and test datasets

The processed datasets are not stored in this repository due to their size.
