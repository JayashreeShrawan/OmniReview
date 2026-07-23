"""Denoising Diffusion Probabilistic Model for embedding refinement."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DIFFUSION_STEPS = 100

#   Diffusion Noise Schedule
def get_beta_schedule(num_steps):
    # Linear variance schedule for the forward diffusion process.
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, num_steps)

betas = get_beta_schedule(NUM_DIFFUSION_STEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

#   DenoisingMLP -- Custom-Built From Scratch
class DenoisingMLP(nn.Module):
    # Denoising network for embedding-space diffusion.
    # Predicts the noise added to concatenated [body, head] embeddings.
    #
    # Inputs: noisy x_t, timestep t, condition embedding
    # Output: predicted noise eps
    def __init__(self, embed_dim, cond_dim, hidden_dim=512):
        super(DenoisingMLP, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + cond_dim + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim * 2)  # Predict noise for body + head
        )
    def forward(self, x_t, t, cond):
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, cond, t_emb], dim=1)
        return self.net(h)

#  Diffusion Forward Process and Loss Function
def diffusion_loss_fn(model, x_0, cond):
    # Score-matching objective: predict noise from noisy input (Topic #12).
    batch_size = x_0.size(0)
    t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size, 1), device=DEVICE).float()
    t_int = t.long().squeeze()

    noise = torch.randn_like(x_0)
    a_t = sqrt_alphas_cumprod[t_int].unsqueeze(1)
    b_t = sqrt_one_minus_alphas_cumprod[t_int].unsqueeze(1)
    x_t = a_t * x_0 + b_t * noise  # Forward diffusion: q(x_t | x_0)

    predicted_noise = model(x_t, t / NUM_DIFFUSION_STEPS, cond)
    return F.mse_loss(predicted_noise, noise)

#   Reverse Diffusion -- Sampling Function
@torch.no_grad()
def p_sample_loop(model, shape, cond):
    # Reverse diffusion: iteratively denoise from pure noise to clean embeddings.
    model.eval()
    batch_size = shape[0]
    x_t = torch.randn(shape, device=DEVICE)

    for i in reversed(range(NUM_DIFFUSION_STEPS)):
        t = torch.full((batch_size, 1), i, device=DEVICE).float()
        noise_pred = model(x_t, t / NUM_DIFFUSION_STEPS, cond)

        alpha_t = alphas[i]
        alpha_t_cumprod = alphas_cumprod[i]
        beta_t = betas[i]

        noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
        x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * noise_pred) + torch.sqrt(beta_t) * noise

    return x_t
