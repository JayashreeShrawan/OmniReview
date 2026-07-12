# Appendix Diagnostics
# Extracted from OmniReview_Colab.ipynb

#  Final Summary Table -- All Components
print("=" * 70)
print(" OmniReview -- Final Component Summary")
print("=" * 70)

component_summary = {
    'TextCNN': {'Type': 'Classifier', 'Topics': '#3', 'Custom': 'Yes', 'Status': 'Trained'},
    'CharRNN': {'Type': 'Generator', 'Topics': '#5,#6', 'Custom': 'Yes', 'Status': 'Trained'},
    'OmniVAE': {'Type': 'Latent Space', 'Topics': '#4,#9', 'Custom': 'Yes', 'Status': 'Trained'},
    'T5+LoRA': {'Type': 'Text Decoder', 'Topics': '#6,#7,#8', 'Custom': 'Wrapper', 'Status': 'Fine-tuned'},
    'Flow (MAF)': {'Type': 'Density Est.', 'Topics': '#10', 'Custom': 'Config', 'Status': 'Trained'},
    'DiffusionMLP': {'Type': 'Refiner', 'Topics': '#12,#13', 'Custom': 'Yes', 'Status': 'Trained'},
    'GAN Disc.': {'Type': 'Quality Gate', 'Topics': '#11', 'Custom': 'Yes', 'Status': 'Trained'},
}

comp_df = pd.DataFrame(component_summary).T
print(comp_df.to_string())

#   List All Saved Outputs
print("=== All Saved Files ===\n")

for dir_path, dir_name in [(CHECKPOINT_DIR, 'Checkpoints'), (RESULTS_DIR, 'Results'),
                            (OUTPUTS_DIR, 'Outputs'), (DATA_DIR, 'Data')]:
    print(f" {dir_name} ({dir_path}):")
    total = 0
    for f in sorted(dir_path.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1e6
            total += size
            print(f"  {f.name:45s}: {size:>8.1f} MB")
    print(f"  {'TOTAL':45s}: {total:>8.1f} MB\n")

#   Reproducibility Verification
print("=== Reproducibility Information ===\n")
print(f"  Random Seed:      {SEED}")
print(f"  PyTorch Version:  {torch.__version__}")
print(f"  CUDA Version:     {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"  Python Version:   {sys.version.split()[0]}")
print(f"  Device Used:      {DEVICE}")
print(f"  Data Categories:  {TARGET_CATEGORIES}")
print(f"  Samples/Group:    {TARGET_SAMPLES_PER_CATEGORY_RATING}")
print(f"\nTo reproduce: Set SEED={SEED}, use same TARGET_CATEGORIES and TARGET_SAMPLES_PER_CATEGORY_RATING")

#   Generate Final Report Data
report = {
    'project': 'OmniReview',
    'total_models': 7,
    'custom_models': 6,
    'dataset': 'Amazon US Customer Reviews',
    'categories': NUM_CATEGORIES,
    'total_samples': len(train_df) + len(val_df) + len(test_df),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'embedding_dim': EMBEDDING_DIM,
    'latent_dim': LATENT_DIM,
    'generated_reviews': len(gen_df),
    'real_coherence': float(real_coherence_mean),
    'pipeline_coherence': float(full_coherence),
    'gan_auc': float(gan_auc)
}
with open(RESULTS_DIR / 'final_report_data.json', 'w') as f:
    json.dump(report, f, indent=2)
print(" Final report data saved")
for k, v in report.items():
    print(f"  {k}: {v}")

#  Final Notebook Cleanup
print(" OmniReview End-to-End Pipeline Execution Complete!")
print(f"\nTotal execution across {7} model architectures.")
print(f"Generated {len(gen_df)} review packages saved to {OUTPUTS_DIR}")
gpu_memory_usage()
free_memory()
print("\n Done! Thank you for reviewing OmniReview! ")

#  Model Parameter Comparison Table
model_params = {
    'TextCNN': sum(p.numel() for p in cnn_model.parameters()),
    'CharRNN': sum(p.numel() for p in rnn_model.parameters()),
    'OmniVAE': sum(p.numel() for p in vae_model.parameters()),
    'T5+LoRA': sum(p.numel() for p in conditioned_t5.parameters()),
    'DenoisingMLP': sum(p.numel() for p in diff_model.parameters()),
    'ReviewDiscriminator': sum(p.numel() for p in gan_discriminator.parameters()),
    'Flow (MAF)': sum(p.numel() for p in flow_model.parameters()),
}

print("=== Model Parameter Comparison ===")
param_df = pd.DataFrame([
    {'Model': k, 'Parameters': v, 'Size (MB)': v * 4 / 1e6}
    for k, v in model_params.items()
])
param_df = param_df.sort_values('Parameters', ascending=False)
print(param_df.to_string(index=False))

total = sum(model_params.values())
print(f"\nTotal parameters across all models: {total:,}")
print(f"Total model size: {total * 4 / 1e6:.1f} MB")

#  Parameter Count Visualization
plt.figure(figsize=(12, 6))
names = param_df['Model'].values
sizes = param_df['Parameters'].values / 1e6  # In millions

bars = plt.barh(names, sizes, color='steelblue', edgecolor='black', alpha=0.8)
plt.xlabel('Parameters (Millions)')
plt.title('Model Parameter Counts', fontsize=14)
for bar, val in zip(bars, sizes):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}M', ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.show()

#  Training Time Summary
print("=== Training Epochs Summary ===")
training_config = {
    'TextCNN': NUM_EPOCHS_CNN,
    'CharRNN': NUM_EPOCHS_RNN,
    'OmniVAE': NUM_EPOCHS_VAE,
    'T5+LoRA': NUM_EPOCHS_TRANSFORMER,
    'Normalizing Flow': NUM_EPOCHS_FLOW,
    'Diffusion': NUM_EPOCHS_DIFFUSION,
    'GAN Disc.': NUM_EPOCHS_GAN,
}
for model, epochs in training_config.items():
    print(f"  {model:20s}: {epochs} epochs")

#  VAE Reconstruction Error per Category
vae_model.eval()
recon_by_cat = {}
with torch.no_grad():
    for batch in test_emb_loader:
        body = batch['body_emb'].to(DEVICE)
        head = batch['head_emb'].to(DEVICE)
        cat = batch['category'].to(DEVICE)
        rat = batch['rating'].to(DEVICE)
        body_rec, head_rec, mu, logvar = vae_model(body, head, cat, rat)

        for i in range(len(cat)):
            c = cat[i].item()
            err = F.mse_loss(body_rec[i], body[i]).item() + F.mse_loss(head_rec[i], head[i]).item()
            if c not in recon_by_cat:
                recon_by_cat[c] = []
            recon_by_cat[c].append(err)

print("=== VAE Reconstruction Error by Category ===")
for c_idx in sorted(recon_by_cat.keys()):
    cat_name = le_cat.inverse_transform([c_idx])[0]
    errors = recon_by_cat[c_idx]
    print(f"  {cat_name:25s}: MSE = {np.mean(errors):.4f} +/- {np.std(errors):.4f}")

#   VAE Reconstruction Error per Rating
recon_by_rating = {}
with torch.no_grad():
    for batch in test_emb_loader:
        body = batch['body_emb'].to(DEVICE)
        head = batch['head_emb'].to(DEVICE)
        cat = batch['category'].to(DEVICE)
        rat = batch['rating'].to(DEVICE)
        body_rec, head_rec, mu, logvar = vae_model(body, head, cat, rat)

        for i in range(len(rat)):
            r = rat[i].item()
            err = F.mse_loss(body_rec[i], body[i]).item() + F.mse_loss(head_rec[i], head[i]).item()
            if r not in recon_by_rating:
                recon_by_rating[r] = []
            recon_by_rating[r].append(err)

print("=== VAE Reconstruction Error by Star Rating ===")
for r_idx in sorted(recon_by_rating.keys()):
    errors = recon_by_rating[r_idx]
    print(f"  {r_idx+1} Stars: MSE = {np.mean(errors):.4f} +/- {np.std(errors):.4f}")

#  Plot VAE Reconstruction Error by Rating
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# By category
cat_names = [le_cat.inverse_transform([c])[0] for c in sorted(recon_by_cat.keys())]
cat_means = [np.mean(recon_by_cat[c]) for c in sorted(recon_by_cat.keys())]
cat_stds = [np.std(recon_by_cat[c]) for c in sorted(recon_by_cat.keys())]
axes[0].bar(cat_names, cat_means, yerr=cat_stds, capsize=5, color='steelblue', edgecolor='black')
axes[0].set_title('VAE Reconstruction MSE by Category')
axes[0].set_ylabel('MSE')
axes[0].tick_params(axis='x', rotation=30)

# By rating
rating_labels = [f'{r+1} Stars' for r in sorted(recon_by_rating.keys())]
rat_means = [np.mean(recon_by_rating[r]) for r in sorted(recon_by_rating.keys())]
rat_stds = [np.std(recon_by_rating[r]) for r in sorted(recon_by_rating.keys())]
axes[1].bar(rating_labels, rat_means, yerr=rat_stds, capsize=5, color='coral', edgecolor='black')
axes[1].set_title('VAE Reconstruction MSE by Star Rating')
axes[1].set_ylabel('MSE')

plt.tight_layout()
plt.show()

#  T5 Validation Loss
conditioned_t5.eval()
total_val_loss = 0
n_batches = 0
with torch.no_grad():
    for batch in val_t5_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        latent_z = batch['latent_z'].to(DEVICE)
        outputs = conditioned_t5(input_ids, attention_mask, latent_z, labels=labels)
        total_val_loss += outputs.loss.item()
        n_batches += 1

val_loss = total_val_loss / max(n_batches, 1)
print(f"T5 Validation Loss: {val_loss:.4f}")
print(f"T5 Validation Perplexity: {np.exp(min(val_loss, 20)):.2f}")

#   Generate Diverse Reviews — Multiple Temperature Settings
conditioned_t5.eval()
print("=== T5 Generation — Temperature Comparison ===\n")
cat_idx = 0
cat_name = le_cat.inverse_transform([cat_idx])[0]
prompt = f"generate review: category {cat_name} rating 5"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
attention_mask = torch.ones_like(input_ids).to(DEVICE)
z = torch.randn(1, LATENT_DIM).to(DEVICE)

for temp in [0.5, 0.7, 1.0, 1.3]:
    with torch.no_grad():
        gen_tokens = conditioned_t5.generate(
            input_ids=input_ids, attention_mask=attention_mask, latent_z=z,
            max_length=128, do_sample=True, temperature=temp, top_k=50, top_p=0.95
        )
    text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    print(f"  Temperature={temp}: {text}\n")

#  GAN Discriminator — Score Distribution by Category
gan_discriminator.eval()
scores_by_cat = {}

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

        scores = gan_discriminator(body, head, helpfulness, cond).cpu().numpy().flatten()
        for i, c in enumerate(cat.cpu().numpy()):
            if c not in scores_by_cat:
                scores_by_cat[c] = []
            scores_by_cat[c].append(scores[i])

plt.figure(figsize=(12, 6))
for c_idx in sorted(scores_by_cat.keys()):
    cat_name = le_cat.inverse_transform([c_idx])[0]
    sns.kdeplot(scores_by_cat[c_idx], label=cat_name, fill=True, alpha=0.3)
plt.title('GAN Discriminator Score Distribution by Category (Real Data)', fontsize=14)
plt.xlabel('Discriminator Score')
plt.legend()
plt.show()

#   Latent Space Distance Matrix Between Categories
print("=== Latent Space Distance Between Category Centroids ===")
z_all, r_all, c_all = extract_latents(vae_model, test_emb_loader)
centroids = {}
for c_idx in range(NUM_CATEGORIES):
    mask = c_all == c_idx
    if mask.sum() > 0:
        centroids[c_idx] = z_all[mask].mean(axis=0)

cat_names = [le_cat.inverse_transform([c])[0] for c in sorted(centroids.keys())]
n = len(cat_names)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

dist_df = pd.DataFrame(dist_matrix, index=cat_names, columns=cat_names)
print(dist_df.round(3).to_string())

plt.figure(figsize=(8, 6))
sns.heatmap(dist_df, annot=True, fmt='.2f', cmap='YlOrRd', square=True)
plt.title('L2 Distance Between Category Centroids in VAE Latent Space', fontsize=13)
plt.tight_layout()
plt.show()

#  Flow — Conditional Log-Likelihood Heatmap
flow_model.eval()
ll_matrix = np.zeros((NUM_CATEGORIES, 5))

with torch.no_grad():
    for c_idx in range(NUM_CATEGORIES):
        for r_idx in range(5):
            n_eval = 200
            cat_t = torch.full((n_eval,), c_idx, dtype=torch.long).to(DEVICE)
            rat_t = torch.full((n_eval,), r_idx, dtype=torch.long).to(DEVICE)
            ctx = torch.cat([cat_emb_flow(cat_t), rat_emb_flow(rat_t)], dim=1)

            mask = (test_df['category_encoded'].values == c_idx) & (test_df['star_rating'].values == r_idx + 1)
            if mask.sum() < 10:
                ll_matrix[c_idx, r_idx] = float('nan')
                continue
            h_vals = torch.tensor(
                test_df[mask]['helpfulness_scaled'].values[:n_eval],
                dtype=torch.float32
            ).unsqueeze(1).to(DEVICE)

            actual_n = min(len(h_vals), n_eval)
            cat_t = cat_t[:actual_n]
            rat_t = rat_t[:actual_n]
            ctx = ctx[:actual_n]
            h_vals = h_vals[:actual_n]

            ll = flow_model.log_prob(inputs=h_vals, context=ctx).mean().item()
            ll_matrix[c_idx, r_idx] = ll

cat_labels = [le_cat.inverse_transform([c])[0] for c in range(NUM_CATEGORIES)]
ll_df = pd.DataFrame(ll_matrix, index=cat_labels, columns=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'])

plt.figure(figsize=(10, 6))
sns.heatmap(ll_df, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Flow Log-Likelihood Heatmap (Category x Rating)', fontsize=14)
plt.tight_layout()
plt.show()

#  Embedding Space Visualization — Generated vs Real (t-SNE)
with Timer("t-SNE: Generated vs Real"):
    n_vis = min(1000, len(test_body_emb), len(full_body))
    combined_emb = np.vstack([test_body_emb[:n_vis], full_body[:n_vis]])
    labels_vis = ['Real'] * n_vis + ['Generated'] * n_vis

    tsne_vis = TSNE(n_components=2, random_state=SEED, perplexity=30)
    vis_2d = tsne_vis.fit_transform(combined_emb)

plt.figure(figsize=(10, 8))
real_mask = np.array(labels_vis) == 'Real'
plt.scatter(vis_2d[real_mask, 0], vis_2d[real_mask, 1], alpha=0.4, s=10, label='Real', color='blue')
plt.scatter(vis_2d[~real_mask, 0], vis_2d[~real_mask, 1], alpha=0.4, s=10, label='Generated', color='red')
plt.title('t-SNE: Real vs Generated Body Embeddings', fontsize=16)
plt.legend()
plt.show()

#  Coherence Analysis — Per Category
print("=== Coherence by Category (Full Pipeline) ===")
for c_idx in range(NUM_CATEGORIES):
    cat_name = le_cat.inverse_transform([c_idx])[0]
    with torch.no_grad():
        n = 100
        cat_t = torch.full((n,), c_idx, dtype=torch.long).to(DEVICE)
        rat_t = torch.randint(0, 5, (n,)).to(DEVICE)
        c_emb = vae_model.cat_emb(cat_t)
        r_emb = vae_model.rat_emb(rat_t)
        cond = torch.cat([c_emb, r_emb], dim=1)
        diff_out = p_sample_loop(diff_model, (n, EMBEDDING_DIM * 2), cond)
        b, h = diff_out.split(EMBEDDING_DIM, dim=1)
        sims = [1 - cosine_distance(b[i].cpu().numpy(), h[i].cpu().numpy()) for i in range(n)]
    print(f"  {cat_name:25s}: coherence = {np.mean(sims):.4f} +/- {np.std(sims):.4f}")

#   Checkpoint File Sizes
print("=== Checkpoint Files ===")
total_ckpt = 0
for f in sorted(CHECKPOINT_DIR.iterdir()):
    if f.is_file():
        size = f.stat().st_size / 1e6
        total_ckpt += size
        print(f"  {f.name:35s}: {size:>8.1f} MB")
print(f"  {'TOTAL':35s}: {total_ckpt:>8.1f} MB")

#  CNN Misclassification Analysis
# Analyze which ratings the CNN confuses most
cnn_model.eval()
misclass = defaultdict(list)

with torch.no_grad():
    for batch in test_emb_loader:
        x = batch['body_emb'].to(DEVICE)
        y = batch['rating']
        logits = cnn_model(x)
        preds = torch.argmax(logits, dim=1).cpu()

        for true, pred in zip(y.numpy(), preds.numpy()):
            if true != pred:
                misclass[f'{true+1}->{pred+1}'].append(1)

print("=== CNN Misclassification Patterns ===")
misclass_counts = {k: len(v) for k, v in misclass.items()}
sorted_mis = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
for pattern, count in sorted_mis[:10]:
    print(f"  {pattern}: {count} instances")

#  Generated Review Diversity Analysis
# Measure diversity of generated reviews using unique n-grams
from collections import Counter

gen_texts = gen_df['text'].tolist()
all_bigrams = []
for text in gen_texts:
    words = text.lower().split()
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    all_bigrams.extend(bigrams)

unique_ratio = len(set(all_bigrams)) / max(len(all_bigrams), 1)
print(f"=== Generated Text Diversity ===")
print(f"  Total bigrams: {len(all_bigrams):,}")
print(f"  Unique bigrams: {len(set(all_bigrams)):,}")
print(f"  Diversity ratio: {unique_ratio:.4f}")
print(f"  Average review length: {np.mean([len(t.split()) for t in gen_texts]):.1f} words")

#  Helpfulness Distribution — Real vs Flow (KS Test)
from scipy.stats import ks_2samp

flow_model.eval()
with torch.no_grad():
    cat_t = torch.randint(0, NUM_CATEGORIES, (2000,)).to(DEVICE)
    rat_t = torch.randint(0, 5, (2000,)).to(DEVICE)
    ctx = torch.cat([cat_emb_flow(cat_t), rat_emb_flow(rat_t)], dim=1)
    flow_samps = flow_model.sample(1, context=ctx).squeeze(1).cpu().numpy().flatten()

real_help = test_df['helpfulness_scaled'].values[:2000]

ks_stat, ks_pval = ks_2samp(real_help, flow_samps)
print(f"=== Kolmogorov-Smirnov Test: Flow vs Real Helpfulness ===")
print(f"  KS Statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pval:.4e}")
print(f"  Interpretation: {'Distributions are similar' if ks_pval > 0.05 else 'Distributions differ significantly'}")

#  Export All Results as JSON
import json
final_export = {
    'project': 'OmniReview',
    'models': list(model_params.keys()),
    'total_parameters': sum(model_params.values()),
    'dataset_categories': TARGET_CATEGORIES,
    'samples_per_group': TARGET_SAMPLES_PER_CATEGORY_RATING,
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'vae_final_loss': float(vae_history.history['loss'][-1]),
    't5_final_loss': float(t5_history.history['train_loss'][-1]),
    'cnn_test_accuracy': float(test_metrics['accuracy']),
    'real_coherence': float(real_coherence_mean),
    'pipeline_coherence': float(full_coherence),
    'gan_auc': float(gan_auc),
    'generated_review_count': len(gen_df),
}
export_path = RESULTS_DIR / 'omnireview_final_export.json'
with open(export_path, 'w') as f:
    json.dump(final_export, f, indent=2)
print(f"Final export saved: {export_path}")
for k, v in final_export.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

#  Complete -- End of Notebook
print("=" * 60)
print(" OmniReview -- Complete")
print("=" * 60)
print(f"  Total models trained: 7")
print(f"  Custom architectures: 6")
print(f"  Generated reviews: {len(gen_df)}")
print(f"  Pipeline coherence: {full_coherence:.4f}")
print(f"  GAN AUC: {gan_auc:.4f}")
print("=" * 60)

