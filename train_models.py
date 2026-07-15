
#  Loss Function and Optimizer
# CrossEntropy for multi-class classification
# L2 Regularization via weight_decay
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=3)
print(" Loss, optimizer, and LR scheduler configured")

#  CNN Training Function
def train_cnn(model, loader, optimizer, criterion):
    # Train the CNN for one epoch.
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        x = batch['body_emb'].to(DEVICE)
        y = batch['rating'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# CNN Evaluation Function
def eval_cnn(model, loader, criterion):
    # Evaluate the CNN on a dataset.
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch['body_emb'].to(DEVICE)
            y = batch['rating'].to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'f1_macro': f1_score(all_labels, all_preds, average='macro')
    }
    return metrics, all_preds, all_labels

#  Execute CNN Training
cnn_history = TrainingHistory("TextCNN")
best_cnn_acc = 0

print(" Starting TextCNN Training...")
for epoch in range(1, NUM_EPOCHS_CNN + 1):
    with Timer(f"Epoch {epoch}"):
        train_loss, train_acc = train_cnn(cnn_model, train_emb_loader, cnn_optimizer, cnn_criterion)
        val_metrics, _, _ = eval_cnn(cnn_model, val_emb_loader, cnn_criterion)

        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        cnn_scheduler.step(val_loss)

        cnn_history.log(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        print(f"  Epoch {epoch}/{NUM_EPOCHS_CNN} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_cnn_acc:
            best_cnn_acc = val_acc
            save_checkpoint(cnn_model, cnn_optimizer, epoch, val_loss, CHECKPOINT_DIR / 'textcnn_best.pt')

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

#  Initialize VAE
vae_model = OmniVAE(embed_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM,
                     num_categories=NUM_CATEGORIES).to(DEVICE)
vae_optimizer = optim.AdamW(vae_model.parameters(), lr=1e-3, weight_decay=1e-5)

vae_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
print(f"OmniVAE Parameters: {vae_params:,}")
print(vae_model)

# KL Annealing Schedule
def get_kl_weight(epoch, total_epochs, max_weight=0.1):
    # Linearly anneal KL weight from 0 to max_weight over first half of training.
    return min(max_weight, (epoch / (total_epochs * 0.5)) * max_weight)

# Visualize the annealing schedule
kl_weights = [get_kl_weight(e, NUM_EPOCHS_VAE) for e in range(1, NUM_EPOCHS_VAE + 1)]
plt.figure(figsize=(10, 4))
plt.plot(range(1, NUM_EPOCHS_VAE + 1), kl_weights, 'o-', linewidth=2)
plt.title('KL Annealing Schedule', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('KL Weight')
plt.grid(True, alpha=0.3)
plt.show()

# VAE Training Loop
vae_history = TrainingHistory("OmniVAE")

print(" Starting VAE Training...")
for epoch in range(1, NUM_EPOCHS_VAE + 1):
    with Timer(f"Epoch {epoch}"):
        vae_model.train()
        train_loss, train_recon, train_kld = 0, 0, 0
        kl_weight = get_kl_weight(epoch, NUM_EPOCHS_VAE)

        for batch in train_emb_loader:
            body = batch['body_emb'].to(DEVICE)
            head = batch['head_emb'].to(DEVICE)
            cat = batch['category'].to(DEVICE)
            rat = batch['rating'].to(DEVICE)

            vae_optimizer.zero_grad()
            body_rec, head_rec, mu, logvar = vae_model(body, head, cat, rat)
            loss, recon, kld = vae_loss_function(body_rec, head_rec, body, head, mu, logvar, kl_weight)
            loss.backward()
            clip_grad_norm_(vae_model.parameters(), 1.0)
            vae_optimizer.step()

            train_loss += loss.item()
            train_recon += recon.item()
            train_kld += kld.item()

        n = len(train_emb_loader)
        vae_history.log(loss=train_loss/n, recon=train_recon/n, kld=train_kld/n, kl_weight=kl_weight)
        print(f"  Epoch {epoch}/{NUM_EPOCHS_VAE} | Loss: {train_loss/n:.4f} | Recon: {train_recon/n:.4f} | KLD: {train_kld/n:.4f} | KL_w: {kl_weight:.4f}")

save_checkpoint(vae_model, vae_optimizer, NUM_EPOCHS_VAE, train_loss/n, CHECKPOINT_DIR / 'vae_best.pt')

# Apply LoRA to T5 (Parameter-Efficient Fine-Tuning)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,                    # Rank
    lora_alpha=32,          # Scaling
    lora_dropout=0.1,       # Regularization (Topic #2)
    target_modules=["q", "v"]  # Apply to attention Q and V
)

lora_t5 = get_peft_model(base_t5, peft_config)
print(" LoRA applied to T5")
lora_t5.print_trainable_parameters()

conditioned_t5 = LatentConditionedT5(lora_t5, LATENT_DIM).to(DEVICE)

# T5 Data Preparation -- Prompts and Labels
def prepare_t5_data(df):
    # Create prompts and labels for T5 fine-tuning.
    prompts, labels = [], []
    for _, row in df.iterrows():
        cat = row['product_category']
        star = int(row['star_rating'])
        head = str(row['clean_headline'])[:50]
        body = str(row['clean_body'])[:200]
        prompts.append(f"generate review: category {cat} rating {star}")
        labels.append(f"Headline: {head} Body: {body}")
    return prompts, labels

with Timer("Preparing T5 data"):
    train_prompts, train_labels = prepare_t5_data(train_df)
    val_prompts, val_labels = prepare_t5_data(val_df)

print(f"Sample prompt: {train_prompts[0]}")
print(f"Sample label:  {train_labels[0][:100]}...")

# Tokenize Data for T5
def tokenize_t5_data(prompts, labels, max_input=32, max_output=128):
    # Tokenize prompts and labels for T5.
    inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_input, return_tensors='pt')
    targets = tokenizer(labels, padding='max_length', truncation=True, max_length=max_output, return_tensors='pt')
    label_ids = targets.input_ids.clone()
    label_ids[label_ids == tokenizer.pad_token_id] = -100  # Ignore pad tokens in loss
    return inputs, label_ids

with Timer("Tokenizing T5 train data"):
    train_t5_inputs, train_t5_labels = tokenize_t5_data(train_prompts, train_labels)

with Timer("Tokenizing T5 val data"):
    val_t5_inputs, val_t5_labels = tokenize_t5_data(val_prompts, val_labels)

print(f"Train input shape: {train_t5_inputs.input_ids.shape}")
print(f"Train label shape: {train_t5_labels.shape}")

# T5 Dataset with VAE Latents
class T5VAE_Dataset(Dataset):
    # Dataset integrating T5 tokenized data with VAE latent vectors.
    def __init__(self, inputs, labels, vae_latents):
        self.input_ids = inputs.input_ids
        self.attention_mask = inputs.attention_mask
        self.labels = labels
        self.latents = torch.tensor(vae_latents, dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'latent_z': self.latents[idx]
        }

# Extract latents for all splits
z_train, _, _ = extract_latents(vae_model, train_emb_loader)
z_val, _, _ = extract_latents(vae_model, val_emb_loader)

train_t5_dataset = T5VAE_Dataset(train_t5_inputs, train_t5_labels, z_train)
val_t5_dataset = T5VAE_Dataset(val_t5_inputs, val_t5_labels, z_val)

train_t5_loader = DataLoader(train_t5_dataset, batch_size=32, shuffle=True)
val_t5_loader = DataLoader(val_t5_dataset, batch_size=32, shuffle=False)
print(f" T5+VAE DataLoaders: Train={len(train_t5_loader)} batches, Val={len(val_t5_loader)} batches")

# T5 Training Loop
t5_optimizer = optim.AdamW(conditioned_t5.parameters(), lr=3e-4)
t5_history = TrainingHistory("T5_LoRA_VAE")

print(" Starting T5 Fine-Tuning...")
for epoch in range(1, NUM_EPOCHS_TRANSFORMER + 1):
    with Timer(f"Epoch {epoch}"):
        conditioned_t5.train()
        total_loss = 0
        for batch in train_t5_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            latent_z = batch['latent_z'].to(DEVICE)

            t5_optimizer.zero_grad()
            outputs = conditioned_t5(input_ids, attention_mask, latent_z, labels=labels)
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(conditioned_t5.parameters(), 1.0)
            t5_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_t5_loader)
        t5_history.log(train_loss=avg_loss)
        print(f"  Epoch {epoch}/{NUM_EPOCHS_TRANSFORMER} | Loss: {avg_loss:.4f}")

save_checkpoint(conditioned_t5, t5_optimizer, NUM_EPOCHS_TRANSFORMER, avg_loss, CHECKPOINT_DIR / 't5_vae_best.pt')

# Initialize Flow Model and Context Embeddings
# Context features: Cat (8) + Rating (8) = 16
flow_model = create_flow(num_layers=FLOW_NUM_LAYERS).to(DEVICE)
cat_emb_flow = nn.Embedding(NUM_CATEGORIES, 8).to(DEVICE)
rat_emb_flow = nn.Embedding(5, 8).to(DEVICE)

flow_params = (sum(p.numel() for p in flow_model.parameters()) +
               sum(p.numel() for p in cat_emb_flow.parameters()) +
               sum(p.numel() for p in rat_emb_flow.parameters()))

flow_optimizer = optim.AdamW(
    list(flow_model.parameters()) + list(cat_emb_flow.parameters()) + list(rat_emb_flow.parameters()),
    lr=1e-3
)
print(f" Flow initialized. Parameters: {flow_params:,}")

#  Flow Training Loop (Maximum Likelihood)
flow_history = TrainingHistory("NormalizingFlow")

print(" Starting Normalizing Flow Training...")
for epoch in range(1, NUM_EPOCHS_FLOW + 1):
    with Timer(f"Epoch {epoch}"):
        flow_model.train()
        total_loss = 0

        for batch in train_emb_loader:
            cat = batch['category'].to(DEVICE)
            rat = batch['rating'].to(DEVICE)
            helpfulness = batch['helpfulness'].to(DEVICE).unsqueeze(1)

            c_emb = cat_emb_flow(cat)
            r_emb = rat_emb_flow(rat)
            context = torch.cat([c_emb, r_emb], dim=1)

            flow_optimizer.zero_grad()
            loss = -flow_model.log_prob(inputs=helpfulness, context=context).mean()
            loss.backward()
            flow_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_emb_loader)
        flow_history.log(train_loss=avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{NUM_EPOCHS_FLOW} | NLL: {avg_loss:.4f}")

save_checkpoint(flow_model, flow_optimizer, NUM_EPOCHS_FLOW, avg_loss, CHECKPOINT_DIR / 'flow_best.pt')

#  Initialize Diffusion Model
diff_model = DenoisingMLP(embed_dim=EMBEDDING_DIM, cond_dim=32).to(DEVICE)
diff_optimizer = optim.AdamW(diff_model.parameters(), lr=1e-4)

# Reuse VAE condition embeddings for consistency
cat_emb_diff = vae_model.cat_emb
rat_emb_diff = vae_model.rat_emb

diff_params = sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
print(f" Diffusion Model initialized. Parameters: {diff_params:,}")

#  Diffusion Training Loop
diff_history = TrainingHistory("DiffusionRefiner")

print(" Starting Diffusion Training...")
for epoch in range(1, NUM_EPOCHS_DIFFUSION + 1):
    with Timer(f"Epoch {epoch}"):
        diff_model.train()
        total_loss = 0

        for batch in train_emb_loader:
            body = batch['body_emb'].to(DEVICE)
            head = batch['head_emb'].to(DEVICE)
            cat = batch['category'].to(DEVICE)
            rat = batch['rating'].to(DEVICE)

            c_emb = cat_emb_diff(cat)
            r_emb = rat_emb_diff(rat)
            cond = torch.cat([c_emb, r_emb], dim=1)
            x_0 = torch.cat([body, head], dim=1)

            diff_optimizer.zero_grad()
            loss = diffusion_loss_fn(diff_model, x_0, cond)
            loss.backward()
            clip_grad_norm_(diff_model.parameters(), 1.0)
            diff_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_emb_loader)
        diff_history.log(train_loss=avg_loss)
        if epoch % 3 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{NUM_EPOCHS_DIFFUSION} | MSE Loss: {avg_loss:.4f}")

save_checkpoint(diff_model, diff_optimizer, NUM_EPOCHS_DIFFUSION, avg_loss, CHECKPOINT_DIR / 'diffusion_best.pt')

#  Initialize GAN Discriminator
gan_discriminator = ReviewDiscriminator(EMBEDDING_DIM, 32).to(DEVICE)
gan_optimizer = optim.AdamW(gan_discriminator.parameters(), lr=2e-4)
gan_criterion = nn.BCELoss()

gan_params = sum(p.numel() for p in gan_discriminator.parameters() if p.requires_grad)
print(f" GAN Discriminator initialized. Parameters: {gan_params:,}")
print(gan_discriminator)

#  GAN Training Loop (Adversarial)
gan_history = TrainingHistory("GAN_Discriminator")

print(" Starting GAN Discriminator Training...")
for epoch in range(1, NUM_EPOCHS_GAN + 1):
    with Timer(f"Epoch {epoch}"):
        gan_discriminator.train()
        vae_model.eval()
        total_loss, correct_real, correct_fake, total = 0, 0, 0, 0

        for batch in train_emb_loader:
            body_real = batch['body_emb'].to(DEVICE)
            head_real = batch['head_emb'].to(DEVICE)
            help_real = batch['helpfulness'].to(DEVICE)
            cat = batch['category'].to(DEVICE)
            rat = batch['rating'].to(DEVICE)
            bs = body_real.size(0)

            c_emb = vae_model.cat_emb(cat)
            r_emb = vae_model.rat_emb(rat)
            cond = torch.cat([c_emb, r_emb], dim=1)

            # Train with real
            gan_optimizer.zero_grad()
            real_labels = torch.ones(bs, 1).to(DEVICE)
            preds_real = gan_discriminator(body_real, head_real, help_real, cond)
            loss_real = gan_criterion(preds_real, real_labels)

            # Train with fake (from VAE)
            with torch.no_grad():
                z = torch.randn(bs, LATENT_DIM).to(DEVICE)
                body_fake, head_fake = vae_model.decoder(z, cond)
                help_fake = torch.rand(bs).to(DEVICE)

            fake_labels = torch.zeros(bs, 1).to(DEVICE)
            preds_fake = gan_discriminator(body_fake, head_fake, help_fake, cond)
            loss_fake = gan_criterion(preds_fake, fake_labels)

            loss = loss_real + loss_fake
            loss.backward()
            gan_optimizer.step()

            total_loss += loss.item()
            correct_real += (preds_real > 0.5).sum().item()
            correct_fake += (preds_fake <= 0.5).sum().item()
            total += bs

        avg_loss = total_loss / len(train_emb_loader)
        acc_real = correct_real / total
        acc_fake = correct_fake / total
        gan_history.log(loss=avg_loss, acc_real=acc_real, acc_fake=acc_fake)

        if epoch % 3 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{NUM_EPOCHS_GAN} | Loss: {avg_loss:.4f} | Real Acc: {acc_real:.4f} | Fake Acc: {acc_fake:.4f}")

save_checkpoint(gan_discriminator, gan_optimizer, NUM_EPOCHS_GAN, avg_loss, CHECKPOINT_DIR / 'gan_best.pt')

