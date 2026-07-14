# OmniVAE (Encoder, Decoder, Full Model, Loss)


#  VAE Encoder Architecture -- Custom-Built From Scratch
class VAEEncoder(nn.Module):
    # Encodes body + headline embeddings + conditions into latent (mu, log sigma^2).
    def __init__(self, input_dim, latent_dim, cond_dim, hidden_dim=512):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2 + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, body_emb, head_emb, cond):
        x = torch.cat([body_emb, head_emb, cond], dim=1)
        h = F.leaky_relu(self.fc1(x))
        h = self.dropout(h)
        h = F.leaky_relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

print(" VAEEncoder defined")

# VAE Decoder Architecture
class VAEDecoder(nn.Module):
    # Decodes latent vector z + conditions into reconstructed body and headline embeddings.
    def __init__(self, latent_dim, output_dim, cond_dim, hidden_dim=512):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc_out_body = nn.Linear(hidden_dim, output_dim)
        self.fc_out_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        return self.fc_out_body(h), self.fc_out_head(h)

print(" VAEDecoder defined")

# OmniVAE -- Full Conditional VAE
class OmniVAE(nn.Module):
    # Conditional VAE mapping text embeddings to a shared latent space.
    def __init__(self, embed_dim=384, latent_dim=128, num_categories=5, num_ratings=5):
        super(OmniVAE, self).__init__()
        self.embed_dim = embed_dim
        self.cat_emb = nn.Embedding(num_categories, 16)
        self.rat_emb = nn.Embedding(num_ratings, 16)
        cond_dim = 32  # 16 + 16
        self.encoder = VAEEncoder(embed_dim, latent_dim, cond_dim)
        self.decoder = VAEDecoder(latent_dim, embed_dim, cond_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, body, head, category, rating):
        c_emb = self.cat_emb(category)
        r_emb = self.rat_emb(rating)
        cond = torch.cat([c_emb, r_emb], dim=1)
        mu, logvar = self.encoder(body, head, cond)
        z = self.reparameterize(mu, logvar)
        body_rec, head_rec = self.decoder(z, cond)
        return body_rec, head_rec, mu, logvar

print(" OmniVAE defined (custom-built from scratch)")

# VAE Loss Function with KL Annealing
def vae_loss_function(body_rec, head_rec, body_orig, head_orig, mu, logvar, kl_weight=1.0):
    # ELBO loss = Reconstruction + KL Divergence  .
    recon_loss_body = F.mse_loss(body_rec, body_orig, reduction='mean')
    recon_loss_head = F.mse_loss(head_rec, head_orig, reduction='mean')
    recon_loss = recon_loss_body + recon_loss_head
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + kl_weight * kld_loss
    return total_loss, recon_loss, kld_loss

print(" VAE loss function defined (ELBO with KL annealing)")

