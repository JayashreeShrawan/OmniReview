# ReviewDiscriminator GAN


#  ReviewDiscriminator -- Custom-Built From Scratch
class ReviewDiscriminator(nn.Module):
    # GAN Discriminator evaluating realism of review embedding packages.
    # Input: body_emb + head_emb + helpfulness_scalar + condition_emb
    # Output: probability of being real (sigmoid)
    def __init__(self, embed_dim, cond_dim, hidden_dim=256):
        super(ReviewDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1 + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, body, head, helpfulness, cond):
        x = torch.cat([body, head, helpfulness.unsqueeze(1), cond], dim=1)
        return self.net(x)
print("ReviewDiscriminator defined (custom-built from scratch)")

