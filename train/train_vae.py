# VAE Training & Latent Visualization


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

# Plot VAE Training Metrics
vae_history.plot(['loss', 'recon', 'kld'])

#  Extract Latent Vectors for Visualization
def extract_latents(model, loader):
    # Extract latent means from the VAE encoder.
    model.eval()
    latents, ratings, categories = [], [], []
    with torch.no_grad():
        for batch in loader:
            body = batch['body_emb'].to(DEVICE)
            head = batch['head_emb'].to(DEVICE)
            cat = batch['category'].to(DEVICE)
            rat = batch['rating'].to(DEVICE)
            c_emb = model.cat_emb(cat)
            r_emb = model.rat_emb(rat)
            cond = torch.cat([c_emb, r_emb], dim=1)
            mu, _ = model.encoder(body, head, cond)
            latents.extend(mu.cpu().numpy())
            ratings.extend(rat.cpu().numpy())
            categories.extend(cat.cpu().numpy())
    return np.array(latents), np.array(ratings), np.array(categories)

z_test, r_test, c_test = extract_latents(vae_model, test_emb_loader)
print(f"Extracted latents shape: {z_test.shape}")

# t-SNE Visualization -- By Star Rating
with Timer("t-SNE computation"):
    idx = np.random.choice(len(z_test), min(5000, len(z_test)), replace=False)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    z_tsne = tsne.fit_transform(z_test[idx])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=r_test[idx], cmap='coolwarm', alpha=0.6, s=10)
plt.colorbar(scatter, label='Star Rating (0=1, 4=5)')
plt.title('VAE Latent Space -- t-SNE by Star Rating', fontsize=16)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# t-SNE Visualization -- By Category
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=c_test[idx], cmap='tab10', alpha=0.6, s=10)
cbar = plt.colorbar(scatter, label='Category')
plt.title('VAE Latent Space -- t-SNE by Product Category', fontsize=16)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# UMAP Visualization (Alternative to t-SNE)
try:
    import umap
    with Timer("UMAP computation"):
        reducer = umap.UMAP(n_components=2, random_state=SEED)
        z_umap = reducer.fit_transform(z_test[idx])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    s1 = axes[0].scatter(z_umap[:, 0], z_umap[:, 1], c=r_test[idx], cmap='coolwarm', alpha=0.6, s=10)
    plt.colorbar(s1, ax=axes[0], label='Star Rating')
    axes[0].set_title('UMAP -- By Star Rating', fontsize=14)

    s2 = axes[1].scatter(z_umap[:, 0], z_umap[:, 1], c=c_test[idx], cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(s2, ax=axes[1], label='Category')
    axes[1].set_title('UMAP -- By Category', fontsize=14)
    plt.tight_layout()
    plt.show()
except ImportError:
    print(" umap-learn not installed. Skipping UMAP visualization.")

# Latent Dimension Distribution Histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flatten()):
    if i < LATENT_DIM:
        ax.hist(z_test[:, i], bins=30, alpha=0.7, color='steelblue', density=True)
        ax.set_title(f'z[{i}]', fontsize=10)
        ax.set_yticks([])
plt.suptitle('Distribution of First 8 Latent Dimensions', fontsize=14)
plt.tight_layout()
plt.show()

# Reconstruction Quality Check
vae_model.eval()
with torch.no_grad():
    batch = next(iter(test_emb_loader))
    body = batch['body_emb'].to(DEVICE)
    head = batch['head_emb'].to(DEVICE)
    cat = batch['category'].to(DEVICE)
    rat = batch['rating'].to(DEVICE)
    body_rec, head_rec, mu, logvar = vae_model(body, head, cat, rat)

# Compute reconstruction cosine similarity
body_sims = [1 - cosine_distance(body[i].cpu().numpy(), body_rec[i].cpu().numpy())
             for i in range(min(20, len(body)))]
head_sims = [1 - cosine_distance(head[i].cpu().numpy(), head_rec[i].cpu().numpy())
             for i in range(min(20, len(head)))]

print("=== Reconstruction Quality (Cosine Similarity) ===")
print(f"  Body:     mean={np.mean(body_sims):.4f}  {np.std(body_sims):.4f}")
print(f"  Headline: mean={np.mean(head_sims):.4f}  {np.std(head_sims):.4f}")

# Latent Interpolation Between Ratings
vae_model.eval()
with torch.no_grad():
    # Get mean latent for 1-star and 5-star
    z_1star = z_test[r_test == 0].mean(axis=0)
    z_5star = z_test[r_test == 4].mean(axis=0)

    # Interpolate
    alphas = np.linspace(0, 1, 7)
    z_interp = np.array([z_1star * (1 - a) + z_5star * a for a in alphas])

    print("=== Latent Space Interpolation (1 -> 5) ===")
    print(f"  z_1star norm: {np.linalg.norm(z_1star):.4f}")
    print(f"  z_5star norm: {np.linalg.norm(z_5star):.4f}")
    print(f"  Distance: {np.linalg.norm(z_5star - z_1star):.4f}")
    for i, a in enumerate(alphas):
        norm = np.linalg.norm(z_interp[i])
        print(f"  alpha={a:.2f}: norm={norm:.4f}")

# VAE Summary
print("=" * 60)
print(" Part 8 -- OmniVAE Summary")
print("=" * 60)
print(f"  Architecture: Encoder(800->512->256->mu,sigma) + Decoder(160->256->512->384)")
print(f"  Latent Dim:   {LATENT_DIM}")
print(f"  Parameters:   {vae_params:,}")
print(f"  KL Annealing: Linear 0->0.1 over first 50% epochs")
print(f"  Final Loss:   {vae_history.history['loss'][-1]:.4f}")
print(f"  Final Recon:  {vae_history.history['recon'][-1]:.4f}")
print(f"  Final KLD:    {vae_history.history['kld'][-1]:.4f}")
print(" Part 8 Complete -- VAE trained, latent space visualized!")
gpu_memory_usage()

