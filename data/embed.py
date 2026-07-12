# OmniReview — S-BERT Embedding Extraction
# Extracted from OmniReview_Colab.ipynb

# Initialize Sentence-BERT Model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
sbert_dim = sbert_model.get_sentence_embedding_dimension()
print(f" Sentence-BERT loaded on {DEVICE}")
print(f"  Model: all-MiniLM-L6-v2")
print(f"  Embedding dimension: {sbert_dim}")
assert sbert_dim == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {sbert_dim}"
gpu_memory_usage()

# Embedding Generation Function with Progress Tracking
def generate_embeddings(df, text_col, batch_size=256):
    # Generate Sentence-BERT embeddings for a DataFrame column.
    texts = df[text_col].astype(str).tolist()
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_emb = sbert_model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(batch_emb.cpu().numpy())

        batch_num = i // batch_size + 1
        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches} ({batch_num/total_batches*100:.0f}%)")

    return np.vstack(embeddings)

print(" Embedding generation function defined")

#  Encode Train Split -- Review Bodies
with Timer("Encoding train review bodies"):
    train_body_emb = generate_embeddings(train_df, 'clean_body')
print(f"Train body embeddings shape: {train_body_emb.shape}")
gpu_memory_usage()

#   Encode Train Split -- Review Headlines
with Timer("Encoding train review headlines"):
    train_head_emb = generate_embeddings(train_df, 'clean_headline')
print(f"Train headline embeddings shape: {train_head_emb.shape}")

# Encode Validation Split
with Timer("Encoding validation split"):
    val_body_emb = generate_embeddings(val_df, 'clean_body')
    val_head_emb = generate_embeddings(val_df, 'clean_headline')
print(f"Validation -- Body: {val_body_emb.shape}, Head: {val_head_emb.shape}")

#  Encode Test Split
with Timer("Encoding test split"):
    test_body_emb = generate_embeddings(test_df, 'clean_body')
    test_head_emb = generate_embeddings(test_df, 'clean_headline')
print(f"Test -- Body: {test_body_emb.shape}, Head: {test_head_emb.shape}")

#  Save Embeddings to Disk
np.save(DATA_DIR / 'train_body_emb.npy', train_body_emb)
np.save(DATA_DIR / 'train_head_emb.npy', train_head_emb)
np.save(DATA_DIR / 'val_body_emb.npy', val_body_emb)
np.save(DATA_DIR / 'val_head_emb.npy', val_head_emb)
np.save(DATA_DIR / 'test_body_emb.npy', test_body_emb)
np.save(DATA_DIR / 'test_head_emb.npy', test_head_emb)

print(" All embeddings saved as .npy arrays")
total_size = sum((DATA_DIR / f).stat().st_size for f in [
    'train_body_emb.npy', 'train_head_emb.npy',
    'val_body_emb.npy', 'val_head_emb.npy',
    'test_body_emb.npy', 'test_head_emb.npy'
])
print(f"  Total embedding storage: {total_size / 1e6:.1f} MB")

#  Embedding Quality Check -- Cosine Similarity Sanity
# Similar reviews should have similar embeddings
from scipy.spatial.distance import cosine as cosine_dist

# Pick 5 random pairs and compute similarity
print("=== Embedding Quality Spot-Check ===")
for i in range(5):
    idx = np.random.randint(0, len(train_body_emb))
    body_emb = train_body_emb[idx]
    head_emb = train_head_emb[idx]
    sim = 1 - cosine_dist(body_emb, head_emb)
    print(f"  Review {idx}: body-headline cosine sim = {sim:.4f}")
    print(f"    Body: {train_df.iloc[idx]['clean_body'][:60]}...")
    print(f"    Head: {train_df.iloc[idx]['clean_headline'][:60]}")
    print()

# PCA Visualization of Body Embeddings
pca = PCA(n_components=2, random_state=SEED)
pca_sample_idx = np.random.choice(len(train_body_emb), min(5000, len(train_body_emb)), replace=False)
pca_result = pca.fit_transform(train_body_emb[pca_sample_idx])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                      c=train_df.iloc[pca_sample_idx]['star_rating'].values - 1,
                      cmap='coolwarm', alpha=0.4, s=8)
plt.colorbar(scatter, label='Star Rating (0=1, 4=5)')
plt.title('PCA of Sentence-BERT Body Embeddings (Colored by Star Rating)', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.show()

#  Embedding Statistics
print("=== Embedding Statistics ===")
for name, emb in [('Train Body', train_body_emb), ('Train Head', train_head_emb),
                   ('Val Body', val_body_emb), ('Test Body', test_body_emb)]:
    print(f"  {name:12s}: shape={emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}, "
          f"min={emb.min():.4f}, max={emb.max():.4f}")

# Average Embedding Similarity by Rating
print("=== Average Body-Headline Cosine Similarity by Star Rating ===")
for rating in range(1, 6):
    mask = train_df['star_rating'].values == rating
    if mask.sum() == 0:
        continue
    body_sub = train_body_emb[mask][:500]
    head_sub = train_head_emb[mask][:500]
    sims = [1 - cosine_dist(b, h) for b, h in zip(body_sub, head_sub)]
    print(f"  {rating} : mean sim = {np.mean(sims):.4f}  {np.std(sims):.4f}")

#  EmbeddedReviewDataset -- PyTorch Dataset with Precomputed Embeddings
class EmbeddedReviewDataset(Dataset):
    # PyTorch Dataset using precomputed Sentence-BERT embeddings.
    def __init__(self, df, body_emb, head_emb):
        self.body_emb = torch.tensor(body_emb, dtype=torch.float32)
        self.head_emb = torch.tensor(head_emb, dtype=torch.float32)
        self.ratings = torch.tensor(df['star_rating'].values - 1, dtype=torch.long)
        self.helpfulness = torch.tensor(df['helpfulness_scaled'].values, dtype=torch.float32)
        self.categories = torch.tensor(df['category_encoded'].values, dtype=torch.long)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'body_emb': self.body_emb[idx],
            'head_emb': self.head_emb[idx],
            'rating': self.ratings[idx],
            'helpfulness': self.helpfulness[idx],
            'category': self.categories[idx]
        }

print(" EmbeddedReviewDataset class defined")

#  Create Embedded DataLoaders
train_emb_dataset = EmbeddedReviewDataset(train_df, train_body_emb, train_head_emb)
val_emb_dataset = EmbeddedReviewDataset(val_df, val_body_emb, val_head_emb)
test_emb_dataset = EmbeddedReviewDataset(test_df, test_body_emb, test_head_emb)

train_emb_loader = DataLoader(train_emb_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_emb_loader = DataLoader(val_emb_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_emb_loader = DataLoader(test_emb_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f" Embedded DataLoaders created")
print(f"  Train: {len(train_emb_loader)} batches")
print(f"  Val:   {len(val_emb_loader)} batches")
print(f"  Test:  {len(test_emb_loader)} batches")

#   Verify Embedding Dimensions
batch = next(iter(train_emb_loader))
print(f"Body embedding batch shape: {batch['body_emb'].shape}")
print(f"Head embedding batch shape: {batch['head_emb'].shape}")
print(f"Rating batch shape:         {batch['rating'].shape}")
assert batch['body_emb'].shape[1] == EMBEDDING_DIM, "Body embedding dimension mismatch!"
assert batch['head_emb'].shape[1] == EMBEDDING_DIM, "Head embedding dimension mismatch!"
print(" All dimensions verified!")

#  Disk Usage Summary
print("=== Disk Usage Summary ===")
total_disk = 0
for f in sorted(DATA_DIR.iterdir()):
    if f.is_file():
        size = f.stat().st_size / 1e6
        total_disk += size
        print(f"  {f.name:45s}: {size:>8.1f} MB")
print(f"  {'TOTAL':45s}: {total_disk:>8.1f} MB")

