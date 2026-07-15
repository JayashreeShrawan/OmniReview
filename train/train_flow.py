# Normalizing Flow Training & Evaluation


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

#  Plot Flow Training Loss
flow_history.plot(['train_loss'])

#  Log-Likelihood per Category
flow_model.eval()
print("=== Log-Likelihood per Category ===")
for cat_idx in range(NUM_CATEGORIES):
    cat_name = le_cat.inverse_transform([cat_idx])[0]
    mask = test_df['category_encoded'].values == cat_idx
    if mask.sum() == 0:
        continue
    help_vals = torch.tensor(test_df[mask]['helpfulness_scaled'].values[:500], dtype=torch.float32).unsqueeze(1).to(DEVICE)
    cat_t = torch.full((len(help_vals),), cat_idx, dtype=torch.long).to(DEVICE)
    rat_t = torch.tensor(test_df[mask]['star_rating'].values[:500] - 1, dtype=torch.long).to(DEVICE)
    ctx = torch.cat([cat_emb_flow(cat_t), rat_emb_flow(rat_t)], dim=1)
    with torch.no_grad():
        ll = flow_model.log_prob(inputs=help_vals, context=ctx).mean().item()
    print(f"  {cat_name:25s}: avg log-likelihood = {ll:.4f}")

# Flow Sampling vs Real -- Overall KDE
flow_model.eval()
with torch.no_grad():
    # Sample helpfulness for 1000 random conditions
    cat_t = torch.randint(0, NUM_CATEGORIES, (1000,)).to(DEVICE)
    rat_t = torch.randint(0, 5, (1000,)).to(DEVICE)
    ctx = torch.cat([cat_emb_flow(cat_t), rat_emb_flow(rat_t)], dim=1)
    flow_samples = flow_model.sample(1, context=ctx).squeeze(1).cpu().numpy()

real_data = test_df['helpfulness_scaled'].values

plt.figure(figsize=(10, 6))
sns.kdeplot(flow_samples.flatten(), label='Flow Generated', fill=True, color='blue', alpha=0.5)
sns.kdeplot(real_data[:2000], label='Real Data', fill=True, color='orange', alpha=0.5)
plt.title('Helpfulness Distribution: Real vs Normalizing Flow', fontsize=16)
plt.xlabel('Scaled Helpfulness')
plt.legend()
plt.show()

#  Flow KDE -- Per Star Rating
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for rat_idx in range(5):
    ax = axes[rat_idx]
    # Real
    mask = test_df['star_rating'].values == (rat_idx + 1)
    real = test_df[mask]['helpfulness_scaled'].values[:500]
    sns.kdeplot(real, label='Real', fill=True, color='orange', alpha=0.4, ax=ax)

    # Flow
    with torch.no_grad():
        cat_t = torch.randint(0, NUM_CATEGORIES, (500,)).to(DEVICE)
        rat_t = torch.full((500,), rat_idx, dtype=torch.long).to(DEVICE)
        ctx = torch.cat([cat_emb_flow(cat_t), rat_emb_flow(rat_t)], dim=1)
        samps = flow_model.sample(1, context=ctx).squeeze(1).cpu().numpy().flatten()
    sns.kdeplot(samps, label='Flow', fill=True, color='blue', alpha=0.4, ax=ax)

    ax.set_title(f'{rat_idx+1}-Star')
    ax.legend(fontsize=8)

plt.suptitle('Flow vs Real Helpfulness Distribution -- Per Star Rating', fontsize=14)
plt.tight_layout()
plt.show()

# Flow Summary
print("=" * 60)
print(" Part 10 -- Normalizing Flow Summary")
print("=" * 60)
print(f"  Architecture: MAF ({FLOW_NUM_LAYERS} coupling layers)")
print(f"  Context:      Category(8d) + Rating(8d) = 16d")
print(f"  Parameters:   {flow_params:,}")
print(f"  Final NLL:    {flow_history.history['train_loss'][-1]:.4f}")

