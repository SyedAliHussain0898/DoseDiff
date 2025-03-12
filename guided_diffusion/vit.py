import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_k, x_v):
        q = self.to_q(x_q)
        k = self.to_k(x_k)
        v = self.to_v(x_v)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.atten = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x_q, x_k, x_v):
        x_q = self.norm(x_q)
        x_k = self.norm(x_k)
        x_v = self.norm(x_v)
        x = self.atten(x_q, x_k, x_v) + x_v
        x = self.ff(x) + x
        return x


class ViT_fusion(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width

        self.to_patch_embedding_q = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_k = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_v = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, heads, dim_head, mlp_dim, dropout)

        self.return_linear = nn.Linear(dim, patch_dim)
        self.reshape = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_height // self.patch_height,
                                 w=self.image_width // self.patch_width, p1=self.patch_height, p2=self.patch_width)

    def forward(self, x_q, x_k, x_v):
        b, c, h, w = x_q.shape
        x_q = self.to_patch_embedding_q(x_q)
        x_k = self.to_patch_embedding_k(x_k)
        x_v = self.to_patch_embedding_v(x_v)

        x_q += self.pos_embedding
        x_k += self.pos_embedding
        x_v += self.pos_embedding
        x_q = self.dropout(x_q)
        x_k = self.dropout(x_k)
        x_v = self.dropout(x_v)

        x = self.transformer(x_q, x_k, x_v)
        x = self.return_linear(x)
        x = self.reshape(x)
        return x


if __name__ == '__main__':
    net = ViT_fusion(image_size=(10, 6), patch_size=(5, 3), dim=1024, heads=1, mlp_dim=2048, channels=256, dim_head=64)
    x_q = torch.rand((1, 256, 10, 6))
    x_k = torch.rand((1, 256, 10, 6))
    x_v = torch.rand((1, 256, 10, 6))
    out = net(x_q, x_k, x_v)
    print(out.size())

class ViTFusionModule(nn.Module):
    """
    Enhanced ViT fusion module for multi-stage diffusion.
    Combines CT and distance map features with stage-aware processing.
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        
        # Position embedding for spatial tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, dim))
        
        # Multi-scale feature extraction (inspired by research results [2][6])
        self.multi_scale_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=ks, padding=ks//2, groups=dim),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.SiLU(),
                nn.BatchNorm2d(dim)
            ) for ks in [1, 3, 5, 7]  # Different kernel sizes for multi-scale
        ])
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Stage-specific feature enhancement
        # Each stage may need different types of features
        self.stage_adaptors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.SiLU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(3)  # Assuming 3 stages
        ])
        
        # Fusion projection
        self.fusion_proj = nn.Conv2d(dim * 2, dim, 1)
    
    def forward(self, ct_feat, dis_feat, stage=None):
        """
        Apply ViT fusion with stage-specific processing.
        
        Args:
            ct_feat: CT features [B, C, H, W]
            dis_feat: Distance map features [B, C, H, W]
            stage: Stage indices (optional) [B]
            
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = ct_feat.shape
        
        # Multi-scale feature extraction
        ct_multi_scale = [module(ct_feat) for module in self.multi_scale_modules]
        dis_multi_scale = [module(dis_feat) for module in self.multi_scale_modules]
        
        # Combine multi-scale features
        ct_enhanced = sum(ct_multi_scale) / len(self.multi_scale_modules)
        dis_enhanced = sum(dis_multi_scale) / len(self.multi_scale_modules)
        
        # Reshape for transformer processing
        ct_tokens = ct_enhanced.flatten(2).transpose(1, 2)  # [B, H*W, C]
        dis_tokens = dis_enhanced.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add position embedding
        ct_tokens = ct_tokens + self.pos_embedding
        dis_tokens = dis_tokens + self.pos_embedding
        
        # Concatenate ct and dis tokens
        tokens = torch.cat([ct_tokens, dis_tokens], dim=1)  # [B, 2*H*W, C]
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            tokens = transformer(tokens)
        
        # Split back into ct and dis components
        ct_tokens, dis_tokens = tokens.chunk(2, dim=1)
        
        # Reshape back to spatial format
        ct_feat = ct_tokens.transpose(1, 2).reshape(B, C, H, W)
        dis_feat = dis_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply stage-specific processing if stage is provided
        if stage is not None:
            fused = torch.cat([ct_feat, dis_feat], dim=1)
            outputs = []
            
            for i in range(B):
                stage_idx = min(stage[i].item(), len(self.stage_adaptors) - 1)
                outputs.append(self.stage_adaptors[stage_idx](fused[i:i+1]))
            
            fused = torch.cat(outputs, dim=0)
        else:
            # Default fusion for all samples
            fused = self.fusion_proj(torch.cat([ct_feat, dis_feat], dim=1))
        
        return fused

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    """
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class MultiScaleFusionModule(nn.Module):
    """
    Multi-Scale Fusion Module for combining features at different scales.
    Inspired by Inception-style architectures from the research.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Branch 1: 1x1 convolution for fine-grained details
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # Branch 2: 3x3 convolution for medium-scale features
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels // 4),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # Branch 3: 5x5 convolution for larger-scale features
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels // 4),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # Branch 4: Max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels // 4)
        )
        
        # Projection to combine all branches
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Process each branch
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate all branches along the channel dimension
        combined = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # Final projection
        return self.proj(combined)
