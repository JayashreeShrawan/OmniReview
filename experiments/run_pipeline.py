# Pipeline Inference Demos


#  Generate -- Electronics 5-Star
print("=== OmniReview Pipeline -- Electronics, 5 ===\n")
results_e5 = generate_omnireview(4, 5, num_samples=3)
for r in results_e5:
    print(f"   {r['star_rating']} | Cat: {r['category']} | Helpful: {r['helpfulness']:.2f} | Quality: {r['gan_quality']:.2f}")
    print(f"   {r['text']}\n")

#  Generate -- Books 1-Star
print("=== OmniReview Pipeline -- Books, 1 ===\n")
results_b1 = generate_omnireview(1, 1, num_samples=3)
for r in results_b1:
    print(f"   {r['star_rating']} | Cat: {r['category']} | Helpful: {r['helpfulness']:.2f} | Quality: {r['gan_quality']:.2f}")
    print(f"   {r['text']}\n")

#  Generate -- All Categories, 3-Star
print("=== OmniReview Pipeline -- All Categories, 3 ===\n")
for cat_idx in range(NUM_CATEGORIES):
    results = generate_omnireview(cat_idx, 3, num_samples=1)
    r = results[0]
    print(f"  [{r['category']:25s}] Quality: {r['gan_quality']:.2f} | {r['text'][:100]}...")

#  Quality Comparison Table
print("=== Generation Quality Comparison ===\n")
comparison_rows = []
for cat_idx in range(min(3, NUM_CATEGORIES)):
    for rating in [1, 3, 5]:
        results = generate_omnireview(cat_idx, rating, num_samples=2)
        for r in results:
            comparison_rows.append(r)

comp_df = pd.DataFrame(comparison_rows)
print(comp_df[['category', 'star_rating', 'helpfulness', 'gan_quality']].to_string(index=False))
print(f"\nMean GAN Quality: {comp_df['gan_quality'].mean():.4f}")

#  Side-by-Side -- RNN Baseline vs OmniReview
print("=== Side-by-Side Comparison: RNN Baseline vs OmniReview ===\n")
for rating in [1, 5]:
    print(f"--- {rating}-Star Reviews ---")
    rnn_text = generate_rnn_text(rnn_model, "the ", rating=rating, length=120, temperature=0.7)
    omni_result = generate_omnireview(0, rating, num_samples=1)[0]

    print(f"  RNN Baseline:  {rnn_text[:120]}")
    print(f"  OmniReview:    {omni_result['text'][:120]}")
    print()

#  Pipeline Timing Analysis
import time

timing = {}
for name, func in [
    ("Flow Sampling", lambda: flow_model.sample(1, context=torch.cat([cat_emb_flow(torch.zeros(10, dtype=torch.long).to(DEVICE)), rat_emb_flow(torch.zeros(10, dtype=torch.long).to(DEVICE))], dim=1))),
    ("VAE Decode", lambda: vae_model.decoder(torch.randn(10, LATENT_DIM).to(DEVICE), torch.randn(10, 32).to(DEVICE))),
]:
    start = time.time()
    for _ in range(10):
        func()
    timing[name] = (time.time() - start) / 10

print("=== Pipeline Component Timing (avg over 10 runs) ===")
for name, t in timing.items():
    print(f"  {name:20s}: {t*1000:.1f} ms")

#  Save Generated Outputs
all_generated = []
for cat_idx in range(NUM_CATEGORIES):
    for rating in [1, 2, 3, 4, 5]:
        results = generate_omnireview(cat_idx, rating, num_samples=2)
        all_generated.extend(results)

gen_df = pd.DataFrame(all_generated)
gen_df.to_csv(OUTPUTS_DIR / 'generated_reviews.csv', index=False)
print(f" Saved {len(gen_df)} generated reviews to {OUTPUTS_DIR / 'generated_reviews.csv'}")

# Also save as readable text
with open(OUTPUTS_DIR / 'samples.txt', 'w', encoding='utf-8') as f:
    for _, row in gen_df.iterrows():
        f.write(f"[{row['category']}, {row['star_rating']}, Q:{row['gan_quality']:.2f}]\n")
        f.write(f"{row['text']}\n\n")
print(f" Saved readable samples to {OUTPUTS_DIR / 'samples.txt'}")

#  Pipeline Summary
print("=" * 60)
print(" Part 13 -- Full Pipeline Summary")
print("=" * 60)
print(f"  Components: VAE -> Flow -> Diffusion -> GAN -> T5")
print(f"  Generated:  {len(gen_df)} review packages")
print(f"  Mean Quality: {gen_df['gan_quality'].mean():.4f}")
print(f"  Outputs saved to: {OUTPUTS_DIR}")

