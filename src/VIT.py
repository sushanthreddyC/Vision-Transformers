import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from Attention_block import AttentionBlock
class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super(VisionTransformer, self).__init__()
        
        self.patch_embedding = nn.Conv2d(num_channels, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim), requires_grad=True)
        
        self.positional_encodings = nn.Parameter(torch.zeros(1+num_patches, 1, embed_dim), requires_grad=True)
        self.transformer_blocks = nn.ModuleList([AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
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

        
        pe = self.positional_encodings[:x.shape[0]]
        x = x + pe

        for transformer_block in self.transformer_blocks :
            x = transformer_block(x)

        cls_op_token = x[0]
        out = self.mlp_head(cls_op_token)
        
        return out