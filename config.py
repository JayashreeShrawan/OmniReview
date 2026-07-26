from pathlib import Path

# Random seed
SEED = 42

# Dataset categories
TARGET_CATEGORIES = [
    "Electronics",
    "Books",
    "Camera",
    "Apparel",
    "Digital_Video_Games"
]

NUM_CATEGORIES = len(TARGET_CATEGORIES)

# Sampling parameters
TARGET_SAMPLES_PER_CATEGORY_RATING = 5000
MIN_TOTAL_VOTES = 5
MAX_REVIEWS_PER_FILE = None

# Data directories
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")

# Training
BATCH_SIZE = 32

# Review preprocessing
MIN_WORDS = 5
MAX_WORDS = 300
