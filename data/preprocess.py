# OmniReview — Data Download, Cleaning & Splitting
# Extracted from OmniReview_Colab.ipynb

import os

os.environ['KAGGLE_USERNAME'] = "Senthilrulz"

os.environ['KAGGLE_KEY'] = "KGAT_c2d6f4388b635e29c2154565d885934a"

# Configure Kaggle API for dataset download
# For Google Colab: Upload your kaggle.json via the Files panel first

import os

# Check multiple possible locations for kaggle.json
kaggle_configured = False

# Option 1: Uploaded directly to /content/
if os.path.exists('/content/kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    os.system('cp /content/kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    kaggle_configured = True
    print(" Kaggle API configured from uploaded file")

# Option 2: Already in ~/.kaggle/
elif os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
    kaggle_configured = True
    print(" Kaggle API already configured")

# Option 3: In current directory
elif os.path.exists('./kaggle.json'):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    os.system('cp ./kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    kaggle_configured = True
    print(" Kaggle API configured from current directory")

if not kaggle_configured:
    print(" Kaggle API not configured!")
    print("  Step 1: Go to https://www.kaggle.com/settings -> API -> Create New Token")
    print("  Step 2: Download kaggle.json")
    print("  Step 3: Upload kaggle.json to the Colab Files panel (left sidebar)")
    print("  Step 4: Re-run this cell")

# Define dataset metadata
# Map category names to their exact Kaggle filenames
DATASET_NAME = "cynthiarempel/amazon-us-customer-reviews-dataset"
# SWAPPED 'Kitchen' to 'Camera'
CATEGORY_FILES = {
    'Electronics':        'amazon_reviews_us_Electronics_v1_00.tsv',
    'Books':              'amazon_reviews_us_Books_v1_02.tsv',
    'Camera':             'amazon_reviews_us_Camera_v1_00.tsv',
    'Apparel':            'amazon_reviews_us_Apparel_v1_00.tsv',
    'Digital_Video_Games': 'amazon_reviews_us_Digital_Video_Games_v1_00.tsv'
}
# Only download categories we need
ACTIVE_CATEGORIES = {k: v for k, v in CATEGORY_FILES.items() if k in TARGET_CATEGORIES}
print(f"Categories to download: {list(ACTIVE_CATEGORIES.keys())}")
print(f"Files: {list(ACTIVE_CATEGORIES.values())}")

# Download dataset files from Kaggle
# Downloads only the specific category files we need

DATA_DIR.mkdir(parents=True, exist_ok=True)

for category, filename in ACTIVE_CATEGORIES.items():
    filepath = DATA_DIR / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1e6
        print(f" {category}: Already downloaded ({size_mb:.0f} MB)")
    else:
        print(f"  Downloading {category} ({filename})...")
        os.system(f'kaggle datasets download -d {DATASET_NAME} -f {filename} -p {DATA_DIR} --force')

        # Kaggle downloads as .zip -- extract
        zip_path = DATA_DIR / f"{filename}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
            zip_path.unlink()
            size_mb = filepath.stat().st_size / 1e6
            print(f" {category}: Downloaded and extracted ({size_mb:.0f} MB)")
        elif filepath.exists():
            size_mb = filepath.stat().st_size / 1e6
            print(f" {category}: Downloaded ({size_mb:.0f} MB)")
        else:
            print(f" {category}: Download failed! Check Kaggle API configuration.")

print(f"\n Data directory contents:")
for f in sorted(DATA_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1e6:.0f} MB)")

# Load and inspect each category file individually
# We load one file at a time to understand the data structure

COLUMNS_TO_LOAD = [
    'product_category', 'star_rating', 'review_headline', 'review_body',
    'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_date'
]

category_stats = {}

for category, filename in ACTIVE_CATEGORIES.items():
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f" {category}: File not found at {filepath}")
        continue

    with Timer(f"Loading {category}"):
        df_temp = pd.read_csv(
            filepath,
            sep='\t',
            on_bad_lines='skip',
            usecols=lambda c: c in COLUMNS_TO_LOAD,
            dtype={'star_rating': 'Int64', 'helpful_votes': 'Int64', 'total_votes': 'Int64'},
            nrows=MAX_REVIEWS_PER_FILE
        )
        category_stats[category] = {
            'total_rows': len(df_temp),
            'columns': list(df_temp.columns),
            'verified_purchase_pct': (df_temp['verified_purchase'] == 'Y').mean() * 100 if 'verified_purchase' in df_temp.columns else 0,
            'avg_rating': df_temp['star_rating'].mean() if 'star_rating' in df_temp.columns else 0,
            'memory_mb': df_temp.memory_usage(deep=True).sum() / 1e6
        }
        print(f"   {category}: {len(df_temp):,} reviews, {df_temp.shape[1]} columns, {category_stats[category]['memory_mb']:.0f} MB")
        del df_temp

free_memory()
print("\n Category Statistics:")
stats_df = pd.DataFrame(category_stats).T
print(stats_df.to_string())

# Load all categories with filtering applied
# We load each file, apply filters immediately, and keep only filtered data to save memory

combined_dfs = []

for category, filename in ACTIVE_CATEGORIES.items():
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f" Skipping {category} -- file not found")
        continue

    with Timer(f"Loading & filtering {category}"):
        df = pd.read_csv(
            filepath,
            sep='\t',
            on_bad_lines='skip',
            usecols=lambda c: c in COLUMNS_TO_LOAD,
            dtype={'star_rating': 'Int64', 'helpful_votes': 'Int64', 'total_votes': 'Int64'},
            nrows=MAX_REVIEWS_PER_FILE
        )
        original_count = len(df)

        # Filter 1: Verified purchase only
        df = df[df['verified_purchase'] == 'Y']

        # Filter 2: Minimum total votes (for meaningful helpfulness)
        df = df[df['total_votes'] >= MIN_TOTAL_VOTES]

        # Filter 3: Drop NaNs in essential columns
        df = df.dropna(subset=['star_rating', 'review_body', 'review_headline', 'product_category'])

        # Compute helpfulness ratio
        df['helpfulness_ratio'] = df['helpful_votes'] / df['total_votes']

        combined_dfs.append(df)
        print(f"  {category}: {original_count:,} -> {len(df):,} after filtering ({len(df)/original_count*100:.1f}% retained)")

df_raw = pd.concat(combined_dfs, ignore_index=True)
del combined_dfs
free_memory()

print(f"\n Combined filtered dataset: {len(df_raw):,} reviews from {df_raw['product_category'].nunique()} categories")

#Inspect raw data structure

print(f"=== Combined Dataset Structure ===")
print(f"Shape: {df_raw.shape}")
print(f"\nColumn names and types:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col:25s} ({df_raw[col].dtype})")

print(f"\nMemory usage: {df_raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")

#Display sample reviews from each category

for category in TARGET_CATEGORIES:
    cat_df = df_raw[df_raw['product_category'] == category]
    if len(cat_df) == 0:
        continue
    print(f"\n{'=' * 70}")
    print(f" {category} -- {len(cat_df):,} reviews | Sample Reviews")
    print(f"{'=' * 70}")
    sample = cat_df[['star_rating', 'review_headline', 'review_body', 'helpful_votes', 'total_votes']].head(3)
    for idx, row in sample.iterrows():
        print(f"\n   {row['star_rating']} stars | Helpful: {row['helpful_votes']}/{row['total_votes']}")
        print(f"   Headline: {str(row['review_headline'])[:80]}")
        print(f"   Review: {str(row['review_body'])[:150]}...")

#Reviews per category -- distribution check

print("=== Reviews Per Category (After Filtering) ===")
cat_counts = df_raw['product_category'].value_counts()
for cat, count in cat_counts.items():
    print(f"  {cat:25s}: {count:>8,} reviews")
print(f"  {'TOTAL':25s}: {len(df_raw):>8,} reviews")

#Data type overview and null analysis

print("=== Data Types & Non-Null Counts ===")
print(df_raw.info())
print(f"\n=== Null Values ===")
null_counts = df_raw.isnull().sum()
null_pcts = (null_counts / len(df_raw) * 100).round(2)
null_df = pd.DataFrame({'Nulls': null_counts, 'Percent': null_pcts})
print(null_df.to_string())

#Basic statistics for numeric columns

print("=== Numeric Column Statistics ===")
numeric_stats = df_raw[['star_rating', 'helpful_votes', 'total_votes', 'helpfulness_ratio']].describe()
print(numeric_stats.to_string())

#Memory profiling -- understand RAM usage

print("=== Memory Profiling ===")
mem_usage = df_raw.memory_usage(deep=True) / 1e6  # MB
for col, mb in mem_usage.items():
    print(f"  {str(col):25s}: {mb:>8.2f} MB")
print(f"  {'TOTAL':25s}: {mem_usage.sum():>8.2f} MB")

#Save raw combined data for future use

raw_save_path = DATA_DIR / 'omnireview_raw_combined.parquet'
df_raw.to_parquet(raw_save_path, index=False)
print(f" Raw combined data saved: {raw_save_path}")
print(f"  File size: {raw_save_path.stat().st_size / 1e6:.1f} MB")
print(f"\n Part 2 Complete -- {len(df_raw):,} reviews loaded from {df_raw['product_category'].nunique()} categories")

#Reload dataset if starting from a fresh kernel
if 'df_raw' not in locals():
    print("Loading data from parquet...")
    df_raw = pd.read_parquet(DATA_DIR / 'omnireview_raw_combined.parquet')
    print(f" Loaded {len(df_raw):,} reviews")
else:
    print(f" Data already in memory: {len(df_raw):,} reviews")

# Define Preprocessing Parameters
MIN_WORDS = 5      # Reviews shorter than this are uninformative
MAX_WORDS = 300    # Reviews longer than this are capped for efficiency

print(f"Preprocessing parameters:")
print(f"  Min review words: {MIN_WORDS}")
print(f"  Max review words: {MAX_WORDS}")
print(f"  Min total votes: {MIN_TOTAL_VOTES}")

#  Apply Length Filters
original_count = len(df_raw)
df_clean = df_raw[
    (df_raw['review_body_words'] >= MIN_WORDS) &
    (df_raw['review_body_words'] <= MAX_WORDS) &
    (df_raw['star_rating'].notna()) &
    (df_raw['review_body'].notna()) &
    (df_raw['review_headline'].notna())
].copy()

removed = original_count - len(df_clean)
print(f"Length filter: {original_count:,} -> {len(df_clean):,} ({removed:,} removed, {removed/original_count*100:.1f}%)")

#  Handle Missing Helpfulness Values
# Impute missing helpfulness_ratio with median for that (category, star_rating) group
median_help = df_clean.groupby(['product_category', 'star_rating'])['helpfulness_ratio'].transform('median')
df_clean['helpfulness_ratio'] = df_clean['helpfulness_ratio'].fillna(median_help)
df_clean['helpfulness_ratio'] = df_clean['helpfulness_ratio'].fillna(0.0)  # Fallback

null_count = df_clean['helpfulness_ratio'].isnull().sum()
print(f" Missing helpfulness_ratio imputed. Remaining nulls: {null_count}")

# Categorical Encoding
le_cat = LabelEncoder()
df_clean['category_encoded'] = le_cat.fit_transform(df_clean['product_category'])

print("Category encoding:")
for cat, idx in zip(le_cat.classes_, le_cat.transform(le_cat.classes_)):
    count = (df_clean['product_category'] == cat).sum()
    print(f"  {idx}: {cat:25s} ({count:,} reviews)")

#  Advanced Text Cleaning Function
def advanced_clean(text):
    # Clean text for model input: lowercase, remove HTML/URLs, normalize whitespace.
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'<.*?>', ' ', text)              # Remove HTML tags
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()        # Normalize whitespace
    return text

# Test
test_input = "This is <b>GREAT</b>!!! Visit http://amazon.com for more    details."
print(f"Input:  {test_input}")
print(f"Output: {advanced_clean(test_input)}")

#  Apply Advanced Cleaning to Review Bodies
with Timer("Cleaning review bodies"):
    df_clean['clean_body'] = df_clean['review_body'].apply(advanced_clean)

print(f"Sample cleaned body: {df_clean['clean_body'].iloc[0][:100]}...")

# Apply Advanced Cleaning to Headlines
df_clean['review_headline'] = df_clean['review_headline'].fillna("no headline")
with Timer("Cleaning review headlines"):
    df_clean['clean_headline'] = df_clean['review_headline'].apply(advanced_clean)

print(f"Sample cleaned headline: {df_clean['clean_headline'].iloc[0][:80]}")

# Validate Cleaning Quality
print("=== Cleaning Quality Check ===")
# Check for empty strings after cleaning
empty_body = (df_clean['clean_body'].str.len() == 0).sum()
empty_head = (df_clean['clean_headline'].str.len() == 0).sum()
print(f"  Empty bodies after cleaning: {empty_body}")
print(f"  Empty headlines after cleaning: {empty_head}")

# Remove any empty entries
if empty_body > 0 or empty_head > 0:
    df_clean = df_clean[(df_clean['clean_body'].str.len() > 0) & (df_clean['clean_headline'].str.len() > 0)]
    print(f"  Removed empty entries. Remaining: {len(df_clean):,}")

# Show length distribution after cleaning
print(f"\nCleaned body length (chars): mean={df_clean['clean_body'].str.len().mean():.0f}, median={df_clean['clean_body'].str.len().median():.0f}")
print(f"Cleaned headline length (chars): mean={df_clean['clean_headline'].str.len().mean():.0f}, median={df_clean['clean_headline'].str.len().median():.0f}")

# Stratified Sampling -- 5,000 per (category, star_rating)
print(f"Before sampling: {len(df_clean):,} reviews")
print(f"Target: {TARGET_SAMPLES_PER_CATEGORY_RATING:,} per (category, star_rating) group")
print(f"Expected total: ~{NUM_CATEGORIES * 5 * TARGET_SAMPLES_PER_CATEGORY_RATING:,}")

df_clean = df_clean.groupby(['product_category', 'star_rating'], group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), TARGET_SAMPLES_PER_CATEGORY_RATING), random_state=SEED)
).reset_index(drop=True)

print(f"\n After stratified sampling: {len(df_clean):,} reviews")

#  Verify Stratification Balance
print("=== Stratification Balance ===")
strat_counts = df_clean.groupby(['product_category', 'star_rating']).size().unstack(fill_value=0)
print(strat_counts.to_string())
print(f"\nMin group size: {strat_counts.values.min()}")
print(f"Max group size: {strat_counts.values.max()}")

# Visualize
strat_counts.plot(kind='bar', figsize=(14, 6), colormap='viridis')
plt.title('Review Count per Category x Star Rating (After Stratified Sampling)', fontsize=14)
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Star Rating')
plt.tight_layout()
plt.show()

# Feature Scaling for Numeric Variables
scaler = StandardScaler()
df_clean['review_body_words'] = df_clean['clean_body'].apply(lambda x: len(x.split()))
df_clean[['length_scaled', 'helpfulness_scaled']] = scaler.fit_transform(
    df_clean[['review_body_words', 'helpfulness_ratio']]
)
print(" Numeric features scaled (StandardScaler applied)")
print(f"  Length: mean={df_clean['length_scaled'].mean():.4f}, std={df_clean['length_scaled'].std():.4f}")
print(f"  Helpfulness: mean={df_clean['helpfulness_scaled'].mean():.4f}, std={df_clean['helpfulness_scaled'].std():.4f}")

# Feature Distribution After Scaling
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_clean['length_scaled'], bins=50, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Scaled Review Length Distribution')
axes[0].set_xlabel('Scaled Length')

sns.histplot(df_clean['helpfulness_scaled'], bins=50, kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('Scaled Helpfulness Distribution')
axes[1].set_xlabel('Scaled Helpfulness')

plt.tight_layout()
plt.show()

#  Train / Val / Test Split (80/10/10) -- Stratified
stratify_key = df_clean['product_category'].astype(str) + "_" + df_clean['star_rating'].astype(str)

print("Splitting dataset (80/10/10) with stratification...")
train_df, temp_df = train_test_split(df_clean, test_size=0.2, stratify=stratify_key, random_state=SEED)

temp_stratify = temp_df['product_category'].astype(str) + "_" + temp_df['star_rating'].astype(str)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_stratify, random_state=SEED)

print(f" Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

#   Verify Split Proportions
print("=== Split Verification ===")
total = len(train_df) + len(val_df) + len(test_df)
print(f"  Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
print(f"  Val:   {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
print(f"  Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)")

# Verify stratification preserved
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df['star_rating'].value_counts(normalize=True).sort_index()
    print(f"\n  {split_name} star distribution: {dict(dist.round(3))}")

#  PyTorch Dataset Class
class AmazonReviewDataset(Dataset):
    # PyTorch Dataset for text-based access to reviews.
    def __init__(self, df):
        self.texts = df['clean_body'].values
        self.headlines = df['clean_headline'].values
        self.ratings = df['star_rating'].values - 1  # 0-indexed for CrossEntropy
        self.helpfulness = df['helpfulness_ratio'].values
        self.categories = df['category_encoded'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'headline': self.headlines[idx],
            'rating': torch.tensor(self.ratings[idx], dtype=torch.long),
            'helpfulness': torch.tensor(self.helpfulness[idx], dtype=torch.float32),
            'category': torch.tensor(self.categories[idx], dtype=torch.long)
        }

print(" AmazonReviewDataset class defined")

# Create Dataset Instances
train_dataset = AmazonReviewDataset(train_df)
val_dataset = AmazonReviewDataset(val_df)
test_dataset = AmazonReviewDataset(test_df)

print(f" Dataset instances created:")
print(f"  Train: {len(train_dataset):,} samples")
print(f"  Val:   {len(val_dataset):,} samples")
print(f"  Test:  {len(test_dataset):,} samples")

#  Create DataLoaders
train_loader_basic = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_basic = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader_basic = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f" DataLoaders created:")
print(f"  Train batches: {len(train_loader_basic)}")
print(f"  Val batches:   {len(val_loader_basic)}")
print(f"  Test batches:  {len(test_loader_basic)}")

#  Verify DataLoader Output
batch = next(iter(train_loader_basic))
print("=== Sample Batch ===")
print(f"  Ratings shape:      {batch['rating'].shape}")
print(f"  Helpfulness shape:  {batch['helpfulness'].shape}")
print(f"  Category shape:     {batch['category'].shape}")
print(f"  Sample text [0]:    {batch['text'][0][:80]}...")
print(f"  Sample headline [0]: {batch['headline'][0][:60]}")
print(f"  Rating [0]:         {batch['rating'][0].item() + 1} stars")
print(f"  Category [0]:       {batch['category'][0].item()}")

#   Save Preprocessed Splits to Disk
train_df.to_parquet(DATA_DIR / 'train_clean.parquet', index=False)
val_df.to_parquet(DATA_DIR / 'val_clean.parquet', index=False)
test_df.to_parquet(DATA_DIR / 'test_clean.parquet', index=False)

print(" Preprocessed splits saved:")
for name in ['train', 'val', 'test']:
    path = DATA_DIR / f'{name}_clean.parquet'
    print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")

# Outlier Analysis -- Review Length After Preprocessing
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Body length
sns.boxplot(data=train_df, x='star_rating', y='review_body_words', ax=axes[0], palette='Set2')
axes[0].set_title('Review Body Length by Rating (Training Set)')
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Word Count')

# Helpfulness
sns.boxplot(data=train_df, x='star_rating', y='helpfulness_ratio', ax=axes[1], palette='coolwarm')
axes[1].set_title('Helpfulness Ratio by Rating (Training Set)')
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Helpfulness Ratio')

plt.tight_layout()
plt.show()

#  Data Quality Summary
print("=" * 60)
print(" Part 4 -- Data Preprocessing Summary")
print("=" * 60)
print(f"  Final dataset size: {len(df_clean):,}")
print(f"  Train/Val/Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"  Categories: {len(le_cat.classes_)} ({', '.join(le_cat.classes_)})")
print(f"  Star ratings: 1-5 (balanced via stratified sampling)")
print(f"  Text cleaned: HTML removed, lowercased, URLs stripped")
print(f"  Features scaled: length and helpfulness standardized")

# Free raw data from memory
del df_raw, sentiment_sample
free_memory()
print("\n Part 4 Complete -- Data ready for model training!")

