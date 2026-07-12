# OmniReview — Exploratory Data Analysis
# Extracted from OmniReview_Colab.ipynb

#Star Rating Distribution -- Overall
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df_raw, x='star_rating', palette='viridis', order=[1,2,3,4,5])
plt.title('Overall Star Rating Distribution', fontsize=16)
plt.xlabel('Star Rating')
plt.ylabel('Count')

# Add count annotations
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.show()

# Print percentage breakdown
print("\nStar Rating Proportions:")
for rating in range(1, 6):
    count = (df_raw['star_rating'] == rating).sum()
    pct = count / len(df_raw) * 100
    print(f"  {rating} : {count:>8,} ({pct:.1f}%)")

#Star Rating Distribution by Product Category
plt.figure(figsize=(14, 7))
sns.countplot(data=df_raw, x='star_rating', hue='product_category', palette='Set2', order=[1,2,3,4,5])
plt.title('Star Rating Distribution by Product Category', fontsize=16)
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Stacked Proportional Bar Chart -- Rating Proportions per Category
rating_proportions = df_raw.groupby('product_category')['star_rating'].value_counts(normalize=True).unstack(fill_value=0)
rating_proportions = rating_proportions.reindex(columns=[1, 2, 3, 4, 5])

ax = rating_proportions.plot(kind='bar', stacked=True, figsize=(12, 6),
                              colormap='RdYlGn', edgecolor='white')
plt.title('Star Rating Proportions by Category', fontsize=16)
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.legend(title='Star Rating', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Helpfulness Ratio Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df_raw, x='helpfulness_ratio', bins=50, kde=True, color='purple')
plt.title('Distribution of Helpfulness Ratio (helpful_votes / total_votes)', fontsize=16)
plt.xlabel('Helpfulness Ratio')
plt.ylabel('Frequency')
plt.axvline(x=df_raw['helpfulness_ratio'].median(), color='red', linestyle='--',
            label=f"Median: {df_raw['helpfulness_ratio'].median():.2f}")
plt.legend()
plt.show()

# Helpfulness by Star Rating (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_raw, x='star_rating', y='helpfulness_ratio', palette='coolwarm', order=[1,2,3,4,5])
plt.title('Helpfulness Ratio vs Star Rating', fontsize=16)
plt.xlabel('Star Rating')
plt.ylabel('Helpfulness Ratio')
plt.show()

# Helpfulness by Category (Boxplot)
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_raw, x='product_category', y='helpfulness_ratio', palette='Set3')
plt.title('Helpfulness Ratio by Product Category', fontsize=16)
plt.xlabel('Product Category')
plt.ylabel('Helpfulness Ratio')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# Helpfulness Percentiles Table
print("=== Helpfulness Ratio Percentiles ===")
percentiles = df_raw.groupby('product_category')['helpfulness_ratio'].describe(
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)
print(percentiles.to_string())

#Text Length Computation (Characters and Words)
df_raw['review_body_len'] = df_raw['review_body'].astype(str).apply(len)
df_raw['review_body_words'] = df_raw['review_body'].astype(str).apply(lambda x: len(x.split()))
df_raw['review_headline_len'] = df_raw['review_headline'].astype(str).apply(len)
df_raw['review_headline_words'] = df_raw['review_headline'].astype(str).apply(lambda x: len(x.split()))

print("Text length statistics:")
print(f"  Body chars    -- mean: {df_raw['review_body_len'].mean():.0f}, median: {df_raw['review_body_len'].median():.0f}")
print(f"  Body words    -- mean: {df_raw['review_body_words'].mean():.0f}, median: {df_raw['review_body_words'].median():.0f}")
print(f"  Headline chars -- mean: {df_raw['review_headline_len'].mean():.0f}, median: {df_raw['review_headline_len'].median():.0f}")
print(f"  Headline words -- mean: {df_raw['review_headline_words'].mean():.0f}, median: {df_raw['review_headline_words'].median():.0f}")

# Review Body Word Count Distribution
plt.figure(figsize=(12, 6))
capped_words = df_raw[df_raw['review_body_words'] < 500]['review_body_words']
sns.histplot(data=capped_words, bins=50, kde=True, color='teal')
plt.title('Distribution of Review Body Length (Words) -- Capped at 500', fontsize=16)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.axvline(x=capped_words.median(), color='red', linestyle='--', label=f'Median: {capped_words.median():.0f}')
plt.legend()
plt.show()

# Headline Length Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df_raw[df_raw['review_headline_words'] < 30], x='review_headline_words', bins=30, kde=True, color='coral')
plt.title('Distribution of Headline Length (Words)', fontsize=16)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Review Length vs Star Rating (Violin Plot)
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_raw[df_raw['review_body_words'] < 500], x='star_rating',
               y='review_body_words', palette='muted', order=[1,2,3,4,5])
plt.title('Review Word Count by Star Rating', fontsize=16)
plt.xlabel('Star Rating')
plt.ylabel('Word Count')
plt.show()

# Review Length vs Helpfulness (Scatter)
plt.figure(figsize=(10, 8))
sample_scatter = df_raw.sample(n=min(10000, len(df_raw)), random_state=SEED)
sns.scatterplot(data=sample_scatter, x='review_body_words', y='helpfulness_ratio',
                alpha=0.3, color='darkorange', s=15)
plt.title('Review Length vs Helpfulness Ratio (10K Sample)', fontsize=16)
plt.xlabel('Review Word Count')
plt.ylabel('Helpfulness Ratio')
plt.xlim(0, 500)
plt.show()

# Correlation
corr, pval = pearsonr(
    df_raw['review_body_words'].clip(upper=500),
    df_raw['helpfulness_ratio']
)
print(f"Pearson correlation (length vs helpfulness): {corr:.4f} (p={pval:.2e})")

# Verified Purchase Analysis
plt.figure(figsize=(10, 6))
vp_rating = df_raw.groupby(['verified_purchase', 'star_rating']).size().reset_index(name='count')
sns.barplot(data=vp_rating, x='star_rating', y='count', hue='verified_purchase', palette='pastel')
plt.title('Star Ratings by Verified Purchase Status', fontsize=16)
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.show()

# Since we filtered to verified_purchase='Y', this should show 100%
print(f"Verified purchase breakdown: {df_raw['verified_purchase'].value_counts().to_dict()}")

#  Vine Review Analysis
if 'vine' in df_raw.columns:
    plt.figure(figsize=(8, 5))
    vine_counts = df_raw['vine'].value_counts()
    plt.pie(vine_counts, labels=vine_counts.index, autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'], startangle=90)
    plt.title('Vine vs Non-Vine Reviews', fontsize=16)
    plt.show()
    print(f"Vine review breakdown: {vine_counts.to_dict()}")
else:
    print("Vine column not available in dataset")

# Correlation Heatmap for Numeric Variables
numeric_cols = ['star_rating', 'helpful_votes', 'total_votes', 'helpfulness_ratio',
                'review_body_words', 'review_headline_words']
available_cols = [c for c in numeric_cols if c in df_raw.columns]
corr_matrix = df_raw[available_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Heatmap -- Numeric Variables', fontsize=16)
plt.tight_layout()
plt.show()

# Text Cleaning Function for NLP EDA
def simple_clean_text(text):
    # Clean text for word cloud and n-gram analysis.
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words and len(w) > 2])

print(" Text cleaning function defined")

# Word Cloud for 5-Star Reviews
samples_per_star = min(5000, len(df_raw))
text_5star = df_raw[df_raw['star_rating'] == 5]['review_body'].sample(
    n=min(samples_per_star, (df_raw['star_rating'] == 5).sum()), random_state=SEED
).apply(simple_clean_text)

wc_5 = WordCloud(width=1000, height=500, background_color='white',
                 colormap='Greens', max_words=150).generate(" ".join(text_5star))
plt.figure(figsize=(14, 7))
plt.imshow(wc_5, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in 5-Star Reviews', fontsize=18)
plt.show()

# Word Cloud for 1-Star Reviews
text_1star = df_raw[df_raw['star_rating'] == 1]['review_body'].sample(
    n=min(samples_per_star, (df_raw['star_rating'] == 1).sum()), random_state=SEED
).apply(simple_clean_text)

wc_1 = WordCloud(width=1000, height=500, background_color='white',
                 colormap='Reds', max_words=150).generate(" ".join(text_1star))
plt.figure(figsize=(14, 7))
plt.imshow(wc_1, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in 1-Star Reviews', fontsize=18)
plt.show()

# Word Clouds for Each Category
fig, axes = plt.subplots(1, len(TARGET_CATEGORIES), figsize=(6 * len(TARGET_CATEGORIES), 5))
if len(TARGET_CATEGORIES) == 1:
    axes = [axes]

for ax, category in zip(axes, TARGET_CATEGORIES):
    cat_text = df_raw[df_raw['product_category'] == category]['review_body'].sample(
        n=min(3000, (df_raw['product_category'] == category).sum()), random_state=SEED
    ).apply(simple_clean_text)

    if len(" ".join(cat_text)) > 0:
        wc = WordCloud(width=400, height=300, background_color='white',
                       max_words=80).generate(" ".join(cat_text))
        ax.imshow(wc, interpolation='bilinear')
    ax.set_title(category, fontsize=14)
    ax.axis('off')

plt.suptitle('Word Clouds by Product Category', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()

# VADER Sentiment Analysis Setup
sia = SentimentIntensityAnalyzer()
print(" VADER Sentiment Analyzer loaded")

# Test with examples
test_sentences = [
    "This product is absolutely amazing! Best purchase ever!",
    "Terrible quality. Broke after one day. Complete waste of money.",
    "It's okay, nothing special but gets the job done."
]
for sent in test_sentences:
    scores = sia.polarity_scores(sent)
    print(f"  [{scores['compound']:+.2f}] {sent[:60]}")

# Calculate Sentiment Polarity for a Sample
# VADER is slow on large datasets, so we sample
sentiment_sample = df_raw.sample(n=min(20000, len(df_raw)), random_state=SEED).copy()

with Timer("Computing VADER sentiment"):
    sentiment_sample['compound_polarity'] = sentiment_sample['review_body'].astype(str).apply(
        lambda x: sia.polarity_scores(x)['compound']
    )

print(f"Sentiment computed for {len(sentiment_sample):,} reviews")
print(f"  Mean polarity: {sentiment_sample['compound_polarity'].mean():.4f}")
print(f"  Std polarity:  {sentiment_sample['compound_polarity'].std():.4f}")

# Sentiment Polarity vs Star Rating
plt.figure(figsize=(12, 6))
sns.boxplot(data=sentiment_sample, x='star_rating', y='compound_polarity',
            palette='coolwarm', order=[1,2,3,4,5])
plt.title('VADER Sentiment Polarity vs Star Rating', fontsize=16)
plt.xlabel('Star Rating')
plt.ylabel('Compound Polarity Score (-1 to +1)')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.show()

# Sentiment Polarity Distribution
plt.figure(figsize=(12, 6))
for star in [1, 3, 5]:
    subset = sentiment_sample[sentiment_sample['star_rating'] == star]['compound_polarity']
    sns.kdeplot(subset, label=f'{star}-Star', fill=True, alpha=0.3)
plt.title('Sentiment Polarity Distribution by Star Rating', fontsize=16)
plt.xlabel('Compound Polarity')
plt.ylabel('Density')
plt.legend()
plt.show()

# Bigram Analysis Function
def plot_top_ngrams(corpus, n=2, top_k=20, title="Top n-grams"):
    # Plot the most frequent n-grams in a corpus.
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english', max_features=5000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    df_freq = pd.DataFrame(words_freq, columns=['Ngram', 'Frequency'])
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_freq, y='Ngram', x='Frequency', palette='magma')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

print(" Ngram analysis function defined")

# Top Bigrams in 5-Star Reviews
five_star_corpus = sentiment_sample[sentiment_sample['star_rating'] == 5]['review_body'].astype(str).head(5000)
plot_top_ngrams(five_star_corpus, n=2, title="Top 20 Bigrams in 5-Star Reviews")

# Top Bigrams in 1-Star Reviews
one_star_corpus = sentiment_sample[sentiment_sample['star_rating'] == 1]['review_body'].astype(str).head(5000)
plot_top_ngrams(one_star_corpus, n=2, title="Top 20 Bigrams in 1-Star Reviews")

# Top Trigrams -- 5-Star vs 1-Star
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax_idx, (star, color) in enumerate([(5, 'Greens'), (1, 'Reds')]):
    corpus = sentiment_sample[sentiment_sample['star_rating'] == star]['review_body'].astype(str).head(5000)
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english', max_features=5000).fit(corpus)
    bag = vec.transform(corpus)
    sums = bag.sum(axis=0)
    freqs = [(word, sums[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:15]
    df_f = pd.DataFrame(freqs, columns=['Trigram', 'Frequency'])
    sns.barplot(data=df_f, y='Trigram', x='Frequency', palette=color, ax=axes[ax_idx])
    axes[ax_idx].set_title(f'Top 15 Trigrams -- {star}-Star Reviews', fontsize=14)

plt.tight_layout()
plt.show()

#  Topic Modeling (LDA) -- Discover Latent Topics
from sklearn.decomposition import LatentDirichletAllocation

lda_corpus = sentiment_sample['review_body'].astype(str).head(10000)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tf = tf_vectorizer.fit_transform(lda_corpus)
print(f" TF vectorization complete. Shape: {tf.shape}")

# Fit LDA Model
lda = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method='online', random_state=SEED)

with Timer("LDA fitting"):
    lda.fit(tf)

print(" LDA Model fitted.")

# Display Discovered Topics
def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}: ", end="")
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

print("=== LDA Topics Discovered ===")
display_topics(lda, tf_vectorizer.get_feature_names_out(), 10)

# Temporal Trends -- Reviews Over Time
if 'review_date' in df_raw.columns:
    df_raw['review_date'] = pd.to_datetime(df_raw['review_date'], errors='coerce')

    plt.figure(figsize=(14, 6))
    time_data = df_raw.dropna(subset=['review_date']).set_index('review_date')
    time_data.resample('ME').size().plot(color='blue', linewidth=2)
    plt.title('Number of Reviews Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Review Count')
    plt.grid(True)
    plt.show()
else:
    print("review_date column not available for temporal analysis")

# Average Star Rating Over Time
if 'review_date' in df_raw.columns:
    plt.figure(figsize=(14, 6))
    time_data = df_raw.dropna(subset=['review_date']).set_index('review_date')
    time_data.resample('ME')['star_rating'].mean().plot(color='crimson', linewidth=2)
    plt.title('Average Star Rating Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Average Star Rating')
    plt.grid(True)
    plt.ylim(1, 5)
    plt.show()

#  Cross-Tabulation -- Category x Rating Counts
print("=== Cross-Tabulation: Category x Star Rating ===")
cross_tab = pd.crosstab(df_raw['product_category'], df_raw['star_rating'], margins=True)
print(cross_tab.to_string())

#  EDA Summary Statistics
print("=" * 60)
print(" Part 3 -- EDA Summary")
print("=" * 60)
print(f"  Total reviews analyzed: {len(df_raw):,}")
print(f"  Categories: {df_raw['product_category'].nunique()}")
print(f"  Star rating distribution: skewed toward 5-star (typical for Amazon)")
print(f"  Mean helpfulness ratio: {df_raw['helpfulness_ratio'].mean():.3f}")
print(f"  Mean review body length: {df_raw['review_body_words'].mean():.0f} words")
print(f"  VADER polarity correlates well with star rating")

