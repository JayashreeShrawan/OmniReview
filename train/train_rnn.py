# CharRNN Training & Generation


# Character-Level Tokenization for RNN
# Character-level keeps vocabulary small and training fast

# Build vocabulary from training data
all_text = " ".join(train_df['clean_body'].astype(str).tolist())
chars = sorted(list(set(all_text)))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
VOCAB_SIZE = len(chars)

print(f"RNN Vocabulary Size: {VOCAB_SIZE} unique characters")
print(f"Sample chars: {chars[:20]}...")

# RNN Dataset Preparation
def encode_text(text):
    # Convert text to character indices.
    return [char2int[ch] for ch in str(text) if ch in char2int]

# Use a subset for the RNN baseline (character-level training is memory-intensive)
rnn_subset_size = min(10000, len(train_df))
rnn_df = train_df.sample(n=rnn_subset_size, random_state=SEED).copy()
rnn_df['encoded'] = rnn_df['clean_body'].apply(encode_text)

SEQ_LENGTH = 100  # Character sequence length for training

def get_rnn_batches(df, batch_size, seq_length):
    # Create batches of (input, target, rating) for RNN training.
    x, y, ratings = [], [], []
    for _, row in df.iterrows():
        encoded = row['encoded']
        if len(encoded) > seq_length:
            idx = np.random.randint(0, len(encoded) - seq_length)
            x.append(encoded[idx : idx + seq_length])
            y.append(encoded[idx + 1 : idx + seq_length + 1])
            ratings.append(row['star_rating'] - 1)

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    ratings = torch.tensor(ratings, dtype=torch.long)
    dataset = TensorDataset(x, y, ratings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

rnn_loader = get_rnn_batches(rnn_df, BATCH_SIZE, SEQ_LENGTH)
print(f"RNN DataLoader: {len(rnn_loader)} batches, seq_length={SEQ_LENGTH}")

# Initialize RNN Model
rnn_model = CharRNN(VOCAB_SIZE).to(DEVICE)
rnn_optimizer = optim.AdamW(rnn_model.parameters(), lr=1e-3)
rnn_criterion = nn.CrossEntropyLoss()

rnn_params = sum(p.numel() for p in rnn_model.parameters() if p.requires_grad)
print(f"CharRNN Parameters: {rnn_params:,}")
print(rnn_model)

# RNN Training Loop
rnn_history = TrainingHistory("LSTM_Baseline")

print(" Starting CharRNN Training...")
for epoch in range(1, NUM_EPOCHS_RNN + 1):
    with Timer(f"Epoch {epoch}"):
        rnn_model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, (x, y, cond) in enumerate(rnn_loader):
            if x.size(0) != BATCH_SIZE:
                continue  # Skip incomplete batches for hidden state consistency

            x, y, cond = x.to(DEVICE), y.to(DEVICE), cond.to(DEVICE)
            hidden = rnn_model.init_hidden(BATCH_SIZE)
            hidden = tuple([each.data for each in hidden])  # Detach

            rnn_optimizer.zero_grad()
            logits, hidden = rnn_model(x, cond, hidden)
            loss = rnn_criterion(logits, y.view(-1))
            loss.backward()
            clip_grad_norm_(rnn_model.parameters(), 5)
            rnn_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        perplexity = np.exp(min(avg_loss, 20))  # Cap to avoid overflow
        rnn_history.log(train_loss=avg_loss, perplexity=perplexity)
        print(f"  Epoch {epoch}/{NUM_EPOCHS_RNN} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

save_checkpoint(rnn_model, rnn_optimizer, NUM_EPOCHS_RNN, avg_loss, CHECKPOINT_DIR / 'charrnn_best.pt')

# Plot RNN Training -- Loss and Perplexity
rnn_history.plot(['train_loss', 'perplexity'])

# RNN Text Generation Function
def generate_rnn_text(model, start_char, rating, length=150, temperature=0.8):
    # Generate text character-by-character using the trained CharRNN.
    model.eval()
    chars_out = [ch for ch in start_char]
    h = model.init_hidden(1)

    #   changed shape from [[rating - 1]] (2D) to [rating - 1] (1D)
    cond = torch.tensor([rating - 1], dtype=torch.long).to(DEVICE)
    # Seed with start character
    x = torch.tensor([[char2int.get(start_char[0], 0)]]).to(DEVICE)
    with torch.no_grad():
        # First process the rest of the start_char sequence to build hidden state
        for ch in start_char[1:]:
            logits, h = model(x, cond, h)
            x = torch.tensor([[char2int.get(ch, 0)]]).to(DEVICE)

        # Now generate new characters
        for _ in range(length):
            logits, h = model(x, cond, h)
            logits = logits[-1, :] / temperature
            probs = F.softmax(logits, dim=0)
            top_ch = torch.multinomial(probs, 1).item()
            chars_out.append(int2char.get(top_ch, '?'))
            x = torch.tensor([[top_ch]]).to(DEVICE)
    return "".join(chars_out)
print("RNN generation function defined")

# Generate 5-Star Review Samples (RNN Baseline)
print("=== RNN Baseline -- 5-Star Generated Reviews ===\n")
for i in range(3):
    text = generate_rnn_text(rnn_model, "this ", rating=5, length=150, temperature=0.7)
    print(f"  Sample {i+1}: {text}\n")

# Generate 1-Star Review Samples (RNN Baseline)
print("=== RNN Baseline -- 1-Star Generated Reviews ===\n")
for i in range(3):
    text = generate_rnn_text(rnn_model, "i ", rating=1, length=150, temperature=0.7)
    print(f"  Sample {i+1}: {text}\n")

# Generate 3-Star Review Samples (RNN Baseline)
print("=== RNN Baseline -- 3-Star Generated Reviews ===\n")
for i in range(3):
    text = generate_rnn_text(rnn_model, "the ", rating=3, length=150, temperature=0.7)
    print(f"  Sample {i+1}: {text}\n")

# Temperature Sweep -- Effect on Generation Diversity
print("=== Temperature Sweep (5-Star, seed='the ') ===\n")
for temp in [0.3, 0.5, 0.7, 1.0, 1.5]:
    text = generate_rnn_text(rnn_model, "the ", rating=5, length=100, temperature=temp)
    print(f"  T={temp:.1f}: {text}\n")

# Store RNN Baseline Results
rnn_samples = []
for rating in [1, 3, 5]:
    for _ in range(3):
        text = generate_rnn_text(rnn_model, "the ", rating=rating, length=120, temperature=0.7)
        rnn_samples.append({'rating': rating, 'text': text})

results_collector.add_samples('LSTM_Baseline', rnn_samples)
results_collector.add_result('LSTM_Baseline', {
    'final_loss': rnn_history.history['train_loss'][-1],
    'final_perplexity': rnn_history.history['perplexity'][-1]
})

print(" Part 7 Complete -- RNN Baseline trained and samples generated!")

