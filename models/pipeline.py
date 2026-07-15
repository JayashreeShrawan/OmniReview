# Full Pipeline Inference Function


#  Full Pipeline Inference Function
@torch.no_grad()
def generate_omnireview(category_idx, rating_val, num_samples=1):
    # End-to-end OmniReview generation pipeline.
    vae_model.eval(); flow_model.eval(); diff_model.eval()
    gan_discriminator.eval(); conditioned_t5.eval()
    cat = torch.full((num_samples,), category_idx, dtype=torch.long).to(DEVICE)
    rat = torch.full((num_samples,), rating_val - 1, dtype=torch.long).to(DEVICE)
    # 1. Condition Embeddings
    c_emb = vae_model.cat_emb(cat)
    r_emb = vae_model.rat_emb(rat)
    cond = torch.cat([c_emb, r_emb], dim=1)
    # 2. Flow: Predict Helpfulness
    flow_ctx = torch.cat([cat_emb_flow(cat), rat_emb_flow(rat)], dim=1)
    helpfulness_scaled = flow_model.sample(1, context=flow_ctx).squeeze(1).squeeze(1)
    # 3. VAE: Sample Latent and Decode
    z = torch.randn(num_samples, LATENT_DIM).to(DEVICE)
    body_vae, head_vae = vae_model.decoder(z, cond)
    # 4. Diffusion: Refine Embeddings
    diff_output = p_sample_loop(diff_model, (num_samples, EMBEDDING_DIM * 2), cond)
    body_diff, head_diff = diff_output.split(EMBEDDING_DIM, dim=1)
    # 5. GAN: Quality Score
    quality_score = gan_discriminator(body_diff, head_diff, helpfulness_scaled, cond)
    # 6. T5: Generate Text
    cat_name = le_cat.inverse_transform([category_idx])[0]
    prompt = f"generate review: category {cat_name} rating {rating_val}"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.expand(num_samples, -1).to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    gen_tokens = conditioned_t5.generate(
        input_ids=input_ids, attention_mask=attention_mask, latent_z=z,
        max_length=128, num_beams=4, early_stopping=True,
        repetition_penalty=2.5, no_repeat_ngram_size=2
    )
    texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    results = []
    for i in range(num_samples):
        results.append({
            'category': cat_name, 'star_rating': rating_val,
            'helpfulness': helpfulness_scaled[i].item(),
            'gan_quality': quality_score[i].item(),
            'text': texts[i]
        })
    return results
print(" Full OmniReview pipeline function defined")

