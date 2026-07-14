# Diffusion Training & Refinement


#  Visualize Noise Schedule
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(betas.cpu().numpy(), linewidth=2)
axes[0].set_title(' (Noise Variance)')
axes[0].set_xlabel('Timestep')

axes[1].plot(alphas_cumprod.cpu().numpy(), linewidth=2, color='orange')
axes[1].set_title(' (Cumulative Signal Retention)')
axes[1].set_xlabel('Timestep')

axes[2].plot(sqrt_one_minus_alphas_cumprod.cpu().numpy(), linewidth=2, color='red')
axes[2].set_title('sqrt(1-) (Noise Level)')
axes[2].set_xlabel('Timestep')

plt.suptitle('Diffusion Noise Schedule', fontsize=14)
plt.tight_layout()
plt.show()

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

#  Plot Diffusion Training Loss
diff_history.plot(['train_loss'])

#  Generate Diffusion-Refined Embeddings
diff_model.eval()
with torch.no_grad():
    # Generate for a batch of 5-star Electronics
    n_samples = 10
    cat_t = torch.zeros(n_samples, dtype=torch.long).to(DEVICE)
    rat_t = torch.full((n_samples,), 4, dtype=torch.long).to(DEVICE)
    c_emb = cat_emb_diff(cat_t)
    r_emb = rat_emb_diff(rat_t)
    cond = torch.cat([c_emb, r_emb], dim=1)

    with Timer("Diffusion sampling"):
        diff_output = p_sample_loop(diff_model, (n_samples, EMBEDDING_DIM * 2), cond)

    body_diff = diff_output[:, :EMBEDDING_DIM]
    head_diff = diff_output[:, EMBEDDING_DIM:]

print(f"Generated embeddings: body={body_diff.shape}, head={head_diff.shape}")
print(f"Body stats: mean={body_diff.mean():.4f}, std={body_diff.std():.4f}")
print(f"Head stats: mean={head_diff.mean():.4f}, std={head_diff.std():.4f}")

#  Diffusion Refinement -- Before vs After Comparison
# Compare VAE direct output vs diffusion-refined output
vae_model.eval()
with torch.no_grad():
    z = torch.randn(n_samples, LATENT_DIM).to(DEVICE)
    body_vae, head_vae = vae_model.decoder(z, cond)

    # Cosine similarity between body and head (coherence measure)
    vae_sims = [1 - cosine_distance(body_vae[i].cpu().numpy(), head_vae[i].cpu().numpy()) for i in range(n_samples)]
    diff_sims = [1 - cosine_distance(body_diff[i].cpu().numpy(), head_diff[i].cpu().numpy()) for i in range(n_samples)]

print("=== Body-Head Coherence: VAE vs Diffusion ===")
print(f"  VAE Direct:       mean={np.mean(vae_sims):.4f}  {np.std(vae_sims):.4f}")
print(f"  Diffusion Refined: mean={np.mean(diff_sims):.4f}  {np.std(diff_sims):.4f}")

#  Diffusion Summary
print("=" * 60)
print(" Part 11 -- Diffusion Model Summary")
print("=" * 60)
print(f"  Architecture: DenoisingMLP (embed_dim*2 + cond + time -> embed_dim*2)")
print(f"  Timesteps:    {NUM_DIFFUSION_STEPS}")
print(f"  Parameters:   {diff_params:,}")
print(f"  Final MSE:    {diff_history.history['train_loss'][-1]:.4f}")

