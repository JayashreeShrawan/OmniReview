# Experiments

## Overview

This folder contains experimental notebooks, training logs, and result summaries for the OmniReview project.

The experiments are designed to evaluate the effectiveness of combining multiple generative AI models for controllable product review generation.

---

## Planned Experiments

### Experiment 1: Baseline Transformer

**Objective**

Evaluate the performance of the pretrained T5-small model without VAE conditioning.

**Metrics**

- Conditional Perplexity
- BERTScore
- Headline–Body Similarity

---

### Experiment 2: VAE + T5

**Objective**

Evaluate whether a shared latent space learned by the VAE improves semantic consistency between generated review headlines and review bodies.

---

### Experiment 3: VAE + T5 + Diffusion

**Objective**

Measure the impact of Diffusion-LM refinement on text fluency and diversity.

---

### Experiment 4: Full OmniReview

**Objective**

Evaluate the complete framework incorporating:

- Variational Autoencoder (VAE)
- Transformer (T5 + LoRA)
- Diffusion Model
- GAN Discriminator
- Normalizing Flow

---

### Experiment 5: Ablation Study

**Objective**

Measure the contribution of each model component by removing one module at a time and comparing overall performance.

---

## Planned Evaluation Metrics

- Conditional Perplexity
- BERTScore
- Headline–Body Similarity
- Rating–Sentiment Agreement
- Helpfulness Calibration Error
- Embedding-space FID
- GAN Discriminator AUC

---

## Experiment Outputs

This folder will include:

- Jupyter notebooks
- Training logs
- Performance summaries
- Plots and visualizations
- Ablation study results

The experiments will be updated as model development progresses.
