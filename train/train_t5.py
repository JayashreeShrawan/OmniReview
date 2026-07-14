# T5 + LoRA Training & Generation

# Load T5-small and Tokenizer
t5_model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
base_t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(DEVICE)

t5_base_params = sum(p.numel() for p in base_t5.parameters())
print(f" Base T5-small loaded")
print(f"  Parameters: {t5_base_params:,}")
print(f"  Hidden dim: {base_t5.config.d_model}")
print(f"  Vocab size: {base_t5.config.vocab_size}")

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

# LoRA Parameter Analysis
total_t5 = sum(p.numel() for p in conditioned_t5.parameters())
trainable_t5 = sum(p.numel() for p in conditioned_t5.parameters() if p.requires_grad)
frozen_t5 = total_t5 - trainable_t5

print("=== T5 + LoRA Parameter Analysis ===")
print(f"  Total parameters:     {total_t5:,}")
print(f"  Trainable (LoRA+proj): {trainable_t5:,} ({trainable_t5/total_t5*100:.2f}%)")
print(f"  Frozen:               {frozen_t5:,} ({frozen_t5/total_t5*100:.2f}%)")
print(f"  Memory saving:        {(1 - trainable_t5/total_t5)*100:.1f}%")

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

#  Plot T5 Training Loss
t5_history.plot(['train_loss'])

# Generate Reviews with T5 (Beam Search)
conditioned_t5.eval()
print("=== T5 Generated Reviews (Beam Search, width=4) ===\n")
for cat_idx, rating in [(0, 5), (1, 1), (2, 3)]:
    cat_name = le_cat.inverse_transform([cat_idx])[0]
    prompt = f"generate review: category {cat_name} rating {rating}"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        gen_tokens = conditioned_t5.generate(
            input_ids=input_ids, attention_mask=attention_mask, latent_z=z,
            max_length=128, num_beams=4, early_stopping=True,
            repetition_penalty=2.5, no_repeat_ngram_size=2
        )
    text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"  [{cat_name}, {rating}]: {text}\n")

# Generate Reviews with Sampling (Temperature-Based)
print("=== T5 Generated Reviews (Sampling, T=0.8) ===\n")
for cat_idx in range(min(3, NUM_CATEGORIES)):
    cat_name = le_cat.inverse_transform([cat_idx])[0]
    for rating in [1, 5]:
        prompt = f"generate review: category {cat_name} rating {rating}"
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            gen_tokens = conditioned_t5.generate(
                input_ids=input_ids, attention_mask=attention_mask, latent_z=z,
                max_length=128, do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                repetition_penalty=2.5, no_repeat_ngram_size=2
            )
        text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"  [{cat_name}, {rating}]: {text}")
    print()

# T5 Summary
print("=" * 60)
print(" Part 9 -- T5 + LoRA Summary")
print("=" * 60)
print(f"  Base Model:     T5-small ({t5_base_params:,} params)")
print(f"  LoRA Rank:      8, Alpha: 32, Target: Q+V")
print(f"  Trainable:      {trainable_t5:,} ({trainable_t5/total_t5*100:.2f}%)")
print(f"  Final Loss:     {t5_history.history['train_loss'][-1]:.4f}")
print(f"  Innovation:     LatentConditionedT5 (VAE z -> prefix token)")
print(" Part 9 Complete -- T5 fine-tuned and generating!")
gpu_memory_usage()

