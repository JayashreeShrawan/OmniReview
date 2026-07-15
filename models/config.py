# OmniReview — Imports, Configuration & Utilities


# Install all required dependencies
# This cell installs packages needed for the entire OmniReview pipeline
!pip install -q transformers datasets accelerate peft      # Transformer fine-tuning + LoRA
!pip install -q sentence-transformers                       # Sentence-BERT embeddings
!pip install -q nflows                                      # Normalizing Flows library
!pip install -q wordcloud                                   # Word cloud visualizations
!pip install -q kaggle                                      # Dataset download from Kaggle
!pip install -q scikit-learn scipy                          # ML utilities
!pip install -q umap-learn                                  # UMAP dimensionality reduction
!pip install -q nltk                                        # NLP text processing
!pip install -q -U torchao                                  # Fixes peft import version mismatch
print(" All dependencies installed successfully!")

# Core Python and data manipulation imports

import os
import sys
import json
import time
import random
import warnings
import functools
import gc
import zipfile
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', '{:.4f}'.format)

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# PyTorch and deep learning imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.nn.utils import clip_grad_norm_, spectral_norm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/ 1e9:.1f} GB")

# Transformers, NLP, and Sentence-BERT imports

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer

print(f"Transformers version: {__import__('transformers').__version__}")
print(" Transformers and NLP libraries loaded")

#Visualization and plotting imports

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'figure.facecolor': 'white'
})

print(" Visualization libraries configured")

# Scikit-learn and evaluation imports

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.spatial.distance import cosine as cosine_distance

print(" ML and evaluation libraries loaded")

# Normalizing Flows library import
import nflows
from nflows import transforms, distributions, flows
from nflows.transforms import (
    MaskedAffineAutoregressiveTransform,
    RandomPermutation,
    CompositeTransform
)
from nflows.distributions import StandardNormal

print("Normalizing Flows library loaded")

# NLTK imports for text analysis

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
print(" NLTK resources loaded")

# Set random seeds for full reproducibility
# This ensures all experiments can be exactly reproduced

SEED = 42

def set_seed(seed=SEED):
    # Set random seed for reproducibility across all libraries.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()
print(f" Random seed set to {SEED} across Python, NumPy, PyTorch, and CUDA")

#Configure compute device (GPU/CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    print("  No GPU detected. Training will be slow.")
    print("  -> Go to Runtime > Change runtime type > GPU (T4)")

# GPU diagnostics -- detailed hardware info
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("=== GPU Diagnostics ===")
    print(f"  Name:           {props.name}")
    print(f"  Compute Cap:    {props.major}.{props.minor}")
    print(f"  Total Memory:   {props.total_memory / 1e9:.2f} GB")
    print(f"  SM Count:       {props.multi_processor_count}")
    print(f"  Max Threads/SM: {props.max_threads_per_multi_processor}")

    # Quick memory benchmark
    test_tensor = torch.randn(1000, 1000, device='cuda')
    print(f"  Test alloc:     {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    del test_tensor
    torch.cuda.empty_cache()
    print("  Connection test passed")
else:
    print("GPU diagnostics skipped -- no GPU available")

# Define project-wide constants and hyperparameters
# 
# CONFIGURATION — Change these values to adjust the pipeline.
# Any one can modify these constants and re-run the notebook.
# 
# === Data Configuration ===
# SWAPPED 'Kitchen' to 'Camera'
TARGET_CATEGORIES = ['Electronics', 'Books', 'Camera', 'Apparel', 'Digital_Video_Games']
NUM_CATEGORIES = len(TARGET_CATEGORIES)
TARGET_SAMPLES_PER_CATEGORY_RATING = 5000   # Per (category, star_rating) group
MIN_REVIEW_LENGTH = 20                       # Minimum characters in review body
MAX_REVIEW_LENGTH = 512                      # Maximum tokens for transformer input
MIN_TOTAL_VOTES = 5                          # Filter reviews with enough votes
MAX_REVIEWS_PER_FILE = None                  # None = load full file; set a number to limit
# === Model Hyperparameters ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS_VAE = 20
NUM_EPOCHS_TRANSFORMER = 5
NUM_EPOCHS_DIFFUSION = 15
NUM_EPOCHS_GAN = 15
NUM_EPOCHS_FLOW = 20
NUM_EPOCHS_CNN = 10
NUM_EPOCHS_RNN = 10
# === Architecture Sizes ===
EMBEDDING_DIM = 384          # Sentence-BERT (all-MiniLM-L6-v2) output dimension
LATENT_DIM = 128             # VAE latent space dimension
HIDDEN_DIM = 256             # General hidden layer size
NUM_DIFFUSION_STEPS = 100    # Diffusion timesteps
FLOW_NUM_LAYERS = 8          # Normalizing Flow coupling layers
# === Paths ===
DATA_DIR = Path('./data')
CHECKPOINT_DIR = Path('./checkpoints')
RESULTS_DIR = Path('./results')
OUTPUTS_DIR = Path('./outputs')
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print(" Project constants defined")
print(f"  Target categories: {TARGET_CATEGORIES}")
print(f"  Samples per (category, rating): {TARGET_SAMPLES_PER_CATEGORY_RATING:,}")
print(f"  Expected total samples: ~{NUM_CATEGORIES * 5 * TARGET_SAMPLES_PER_CATEGORY_RATING:,}")
print(f"  Embedding dim: {EMBEDDING_DIM}, Latent dim: {LATENT_DIM}")

#Utility functions -- timing and memory monitoring
class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name=""):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f" {self.name}: {self.elapsed:.2f}s")
def gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9

        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory -- Allocated: {allocated:.2f} GB | Cached: {cached:.2f} GB | Total: {total:.2f} GB")
    else:
        print("No GPU available")
def free_memory():
    """Free unused GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print(" Utility functions defined: Timer, gpu_memory_usage, free_memory")

#Checkpoint save and load functions

def save_checkpoint(model, optimizer, epoch, loss, path):
    # Save model checkpoint with optimizer state.
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f" Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    # Load model checkpoint.
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f" Checkpoint loaded: {path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return checkpoint['epoch'], checkpoint['loss']

print(" Checkpoint functions defined: save_checkpoint, load_checkpoint")

#Training history tracker for all models

class TrainingHistory:
    # Track and visualize training metrics across epochs for any model.

    def __init__(self, model_name):
        self.model_name = model_name
        self.history = defaultdict(list)

    def log(self, **kwargs):
        # Log metrics for current epoch. Usage: history.log(loss=0.5, acc=0.9)
        for key, value in kwargs.items():
            self.history[key].append(value)

    def plot(self, metrics=None, figsize=(14, 5)):
        # Plot training curves for specified metrics.
        metrics = metrics or list(self.history.keys())
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
        if n == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            values = self.history[metric]
            ax.plot(range(1, len(values) + 1), values, 'o-', linewidth=2, markersize=4)
            ax.set_title(f'{self.model_name} -- {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def best(self, metric, mode='min'):
        # Return best epoch and value for a metric.
        values = self.history[metric]
        func = min if mode == 'min' else max
        best_val = func(values)
        best_epoch = values.index(best_val) + 1
        return best_epoch, best_val

print(" TrainingHistory class defined")

#Global results collector for final comparison

class ResultsCollector:
    # Collect results from all models for final comparison and ablation study.

    def __init__(self):
        self.results = {}
        self.generated_samples = {}

    def add_result(self, model_name, metrics_dict):
        # Store evaluation metrics for a model.
        self.results[model_name] = metrics_dict
        print(f" Results stored for: {model_name}")

    def add_samples(self, model_name, samples):
        # Store generated review samples.
        self.generated_samples[model_name] = samples

    def summary_table(self):
        # Return a DataFrame comparing all models.
        return pd.DataFrame(self.results).T

    def ablation_table(self):
        # Show ablation results -- full pipeline vs. each component removed.
        if 'OmniReview_Full' in self.results:
            full = self.results['OmniReview_Full']
            rows = []
            for name, metrics in self.results.items():
                delta = {k: metrics.get(k, 0) - full.get(k, 0) for k in full}
                rows.append({'Configuration': name, **metrics, **{f'_{k}': v for k, v in delta.items()}})
            return pd.DataFrame(rows).set_index('Configuration')
        return self.summary_table()

# Initialize global collector
results_collector = ResultsCollector()
print(" ResultsCollector initialized -- will gather metrics from all models")

#Validate environment -- check all critical imports work
import importlib.metadata
env_checks = {
    'torch': torch.__version__,
    'transformers': __import__('transformers').__version__,
    'peft': __import__('peft').__version__,
    'sentence_transformers': __import__('sentence_transformers').__version__,
    'nflows': importlib.metadata.version('nflows'),  # Using importlib to get version
    'sklearn': __import__('sklearn').__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
}
print("=== Environment Validation ===")
all_ok = True
for pkg, ver in env_checks.items():
    status = "[OK]" if ver else "[ERR]"
    print(f"  {status} {pkg}: {ver}")
    if not ver:
        all_ok = False
if all_ok:
    print("\nAll packages verified!")
else:
    print("\nSome packages missing -- check installation")

#Display full environment summary

print("=" * 60)
print(" OmniReview -- Environment Summary")
print("=" * 60)
print(f"  Python:        {sys.version.split()[0]}")
print(f"  PyTorch:       {torch.__version__}")
print(f"  Transformers:  {__import__('transformers').__version__}")
print(f"  Device:        {DEVICE}")
print(f"  Seed:          {SEED}")
print(f"  Categories:    {TARGET_CATEGORIES}")
print(f"  Samples/Group: {TARGET_SAMPLES_PER_CATEGORY_RATING:,}")
print(f"  Embedding Dim: {EMBEDDING_DIM}")
print(f"  Latent Dim:    {LATENT_DIM}")
print(f"  Batch Size:    {BATCH_SIZE}")
print("=" * 60)
print(" Part 1 Complete -- Environment ready!")
gpu_memory_usage()

