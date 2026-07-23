"""TextCNN Sentiment Classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F

#  TextCNN Architecture -- Custom-Built From Scratch
class TextCNN(nn.Module):
    # 1D Convolutional Neural Network for text classification.
    # Treats the SBERT embedding vector as a 1D sequence and applies
    # multiple filter sizes to capture different feature scales.
    #
    # Architecture:
    # - 3 parallel Conv1d branches (filter sizes 2, 3, 4)
    # - Max-pool over each branch
    # - Concatenate -> Dropout -> Linear -> 5-class output
    def __init__(self, embed_dim, num_classes=5, filter_sizes=[2, 3, 4], num_filters=128, dropout_rate=0.5):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)  # Regularization
    def forward(self, x):
        # x shape: [batch, embed_dim] -> [batch, 1, embed_dim]
        x = x.unsqueeze(1)
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))                                    # [batch, num_filters, L]
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2) # [batch, num_filters]
            pooled_outputs.append(p)
        h_pool = torch.cat(pooled_outputs, dim=1)  # [batch, num_filters * 3]
        h_drop = self.dropout(h_pool)
        logits = self.fc(h_drop)
        return logits
