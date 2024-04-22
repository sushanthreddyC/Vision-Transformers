import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from utils import *
from ConvFormer_block import ConvFormer_block
class ConvFormer_VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):

        super(ConvFormer_VisionTransformer, self).__init__()
        
        self.patch_embedding = nn.Conv2d(num_channels, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim), requires_grad=True)
        
        self.positional_encodings = nn.Parameter(torch.zeros(1+num_patches, 1, embed_dim), requires_grad=True)
        self.transformer_blocks = nn.ModuleList([ConvFormer_block(embed_dim, hidden_dim, dropout) for _ in range(1)])
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        
        x = self.patch_embedding(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)

        cls_token_emb = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])

        # print("shape of input", x.shape)
        pe = self.positional_encodings[:x.shape[0]]
        x = x + pe

        for transformer_block in self.transformer_blocks :
            x = transformer_block(x)

        cls_op_token = x[0]
        out = self.mlp_head(cls_op_token)
        
        return out