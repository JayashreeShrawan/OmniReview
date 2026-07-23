"""LatentConditionedT5 - Custom T5 wrapper with VAE latent prefix injection."""
import torch
import torch.nn as nn

# LatentConditionedT5 -- Custom Wrapper (Our Novel Architecture)
class LatentConditionedT5(nn.Module):
    # T5 wrapper that injects a continuous VAE latent vector as a prefix token.
    # This is our custom architecture connecting the VAE latent space to T5.
    def __init__(self, t5_model, latent_dim):
        super().__init__()
        self.t5 = t5_model
        self.latent_dim = latent_dim
        self.latent_proj = nn.Linear(latent_dim, self.t5.config.d_model)
    def forward(self, input_ids, attention_mask, latent_z, labels=None):
        device = input_ids.device
        inputs_embeds = self.t5.shared(input_ids)
        z_embeds = self.latent_proj(latent_z).unsqueeze(1)
        combined_embeds = torch.cat([z_embeds, inputs_embeds], dim=1)
        z_mask = torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([z_mask, attention_mask], dim=1)
        outputs = self.t5(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels)
        return outputs
    def generate(self, input_ids, attention_mask, latent_z, **kwargs):
        device = input_ids.device
        inputs_embeds = self.t5.shared(input_ids)
        z_embeds = self.latent_proj(latent_z).unsqueeze(1)
        combined_embeds = torch.cat([z_embeds, inputs_embeds], dim=1)
        z_mask = torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([z_mask, attention_mask], dim=1)
        return self.t5.generate(inputs_embeds=combined_embeds, attention_mask=combined_mask, **kwargs)
