# Preliminary Experiments

The following preliminary experiments were conducted to validate the feasibility of the proposed OmniReview framework:

- Collected and preprocessed the Amazon US Customer Reviews dataset by removing missing values, filtering verified purchases, and selecting relevant product categories.
- Generated train, validation, and test splits to support model development and evaluation.
- Generated semantic embeddings for review headlines and review bodies using Sentence Transformers.
- Implemented the initial architectures for all five generative models:
  - Variational Autoencoder (VAE)
  - Transformer (T5 + LoRA)
  - Diffusion Model
  - Generative Adversarial Network (GAN)
  - Normalizing Flow
- Verified that each model can be integrated within the proposed OmniReview architecture.
- Established the repository structure, documentation, and training framework to support future model training and evaluation.

These preliminary experiments demonstrate the feasibility of the proposed multi-model framework and provide the foundation for subsequent training, integration, and performance evaluation.
