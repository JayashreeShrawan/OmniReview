# Training

This directory contains training scripts, training configurations, and related resources for the OmniReview project.

The training process includes:

- Training the Variational Autoencoder (VAE)
- Fine-tuning the Transformer (T5 + LoRA)
- Training the Diffusion Model
- Training the GAN components
- Training the Normalizing Flow model

Model-specific training scripts and configurations will be organized within this directory as the project progresses.

## Training Workflow

1. Preprocess the dataset.
2. Create train, validation, and test splits.
3. Train individual generative models.
4. Fine-tune model hyperparameters.
5. Save trained model checkpoints.
6. Evaluate model performance using the evaluation pipeline.
