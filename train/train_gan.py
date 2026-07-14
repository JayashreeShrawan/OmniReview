# GAN Discriminator Training & Evaluation


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

#   Plot GAN Training -- Loss
gan_history.plot(['loss'])

#   Plot GAN Training -- Real vs Fake Accuracy
gan_history.plot(['acc_real', 'acc_fake'])

#   GAN Discriminator Score Distribution
gan_discriminator.eval()
real_scores, fake_scores = [], []

with torch.no_grad():
    for batch in test_emb_loader:
        body = batch['body_emb'].to(DEVICE)
        head = batch['head_emb'].to(DEVICE)
        helpfulness = batch['helpfulness'].to(DEVICE)
        cat = batch['category'].to(DEVICE)
        rat = batch['rating'].to(DEVICE)
        c_emb = vae_model.cat_emb(cat)
        r_emb = vae_model.rat_emb(rat)
        cond = torch.cat([c_emb, r_emb], dim=1)

        real_scores.extend(gan_discriminator(body, head, helpfulness, cond).cpu().numpy().flatten())

        z = torch.randn(body.size(0), LATENT_DIM).to(DEVICE)
        bf, hf = vae_model.decoder(z, cond)
        hfake = torch.rand(body.size(0)).to(DEVICE)
        fake_scores.extend(gan_discriminator(bf, hf, hfake, cond).cpu().numpy().flatten())

plt.figure(figsize=(10, 6))
sns.kdeplot(real_scores, label='Real', fill=True, color='green', alpha=0.5)
sns.kdeplot(fake_scores, label='Generated (VAE)', fill=True, color='red', alpha=0.5)
plt.title('GAN Discriminator Score Distribution', fontsize=16)
plt.xlabel('Discriminator Score (0=Fake, 1=Real)')
plt.legend()
plt.show()

#   GAN AUC Computation
from sklearn.metrics import roc_auc_score
all_scores = real_scores + fake_scores
all_labels_gan = [1] * len(real_scores) + [0] * len(fake_scores)
gan_auc = roc_auc_score(all_labels_gan, all_scores)
print(f"GAN Discriminator AUC: {gan_auc:.4f}")
print(f"  (Ideal for generative quality: AUC -> 0.5 means generator fools discriminator)")

#   GAN Summary
print("=" * 60)
print(" Part 12 -- GAN Discriminator Summary")
print("=" * 60)
print(f"  Architecture: MLP (801->256->128->1) with LeakyReLU + Dropout")
print(f"  Parameters:   {gan_params:,}")
print(f"  Final Loss:   {gan_history.history['loss'][-1]:.4f}")
print(f"  Real Acc:     {gan_history.history['acc_real'][-1]:.4f}")
print(f"  Fake Acc:     {gan_history.history['acc_fake'][-1]:.4f}")
print(f"  AUC:          {gan_auc:.4f}")

