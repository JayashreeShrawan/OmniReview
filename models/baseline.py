# CharRNN (LSTM) Baseline Generator
# Extracted from OmniReview_Colab.ipynb

# Conditional LSTM Architecture -- Custom-Built From Scratch
class CharRNN(nn.Module):
    # Conditional Character-Level LSTM for review generation.
    # Concatenates character embeddings with star-rating condition embeddings
    # at each timestep.
    #
    # Architecture: Embedding -> LSTM(2 layers) -> FC -> vocab_size logits
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2, n_classes=5):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cond_embedding = nn.Embedding(n_classes, embed_dim)  # Rating condition
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, n_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, cond, hidden):
        emb = self.embedding(x)
        cond_emb = self.cond_embedding(cond)
        cond_emb = cond_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        lstm_in = torch.cat([emb, cond_emb], dim=2)
        out, hidden = self.lstm(lstm_in, hidden)
        out = out.reshape(-1, self.hidden_dim)
        logits = self.fc(out)
        return logits, hidden
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(DEVICE),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(DEVICE))
        return hidden
print("CharRNN architecture defined (custom-built from scratch)")

