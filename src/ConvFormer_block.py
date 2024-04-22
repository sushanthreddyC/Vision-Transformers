import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from utils import *

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.unsqueeze(0) 
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = x.squeeze(0)
        return x
    
class ConvFormer_block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, expansion_ratio=2):
        super(ConvFormer_block, self).__init__()
        med_channels = int(expansion_ratio * embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sepconv = SepConv(embed_dim, expansion_ratio)

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
        
        x_norm = self.sepconv(x_norm)
        x = x + x_norm
        
        x_norm = self.ln2(x)
        x_norm = self.ffn(x_norm)
        x = x + x_norm
        return x
        