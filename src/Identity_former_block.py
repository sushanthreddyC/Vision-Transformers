import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
class IdentityFormer_block(nn.Module):
    def __init__(self,embed_dim, hidden_dim, dropout=0.0):
        super(IdentityFormer_block, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.identity = nn.Identity(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x_norm = self.ln1(x)
        x_norm = self.identity(x_norm)
        x = x+x_norm

        x_norm = self.ln2(x)
        x_ffn  = self.ffn(x_norm)
        x = x+x_ffn

        return x