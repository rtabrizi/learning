###############################################################################
#                                                                             #
#   Vision Transformer inspired and guided by labml.ai' implementation        #
#                                                                             #
###############################################################################

import torch
from torch import nn

class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, embed_size: int, in_channels: int, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        # patch + projection \equiv conv where kernel_size, stride = patch_size, patch_size
        self.conv = nn.conv2d(in_channels, out_channels=embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Shape '(batch_size, in_channels, img_heigh, img_width)'

        Returns
        -------
        torch.Tensor
            Shape '(batch_size, num_patches, embed_dim)'
        
        """
        x = self.conv(x) #(batch_size, embed_dim, num_patches **0.5, num_patches **0.5)
        batch_size, embed_dim, p1, p2 = x.shape
        x = x.view(batch_size, embed_dim, p1 * p2)
        return x
    
class Attention(nn.Module):
    """Attention mechanism.
    
    Parameters
    ----------
    dim : int
        embedding dimension, as well as output dimensino of each token and sometimes the hidden state size
    
    num_heads: int
        the number of attention heads.

    qkv_bias : bool
        Whether or not to include bias to the query, key, and value projections.

    attn_p : float
        Dropout probability applied to the query, key, and value tensors

    proj_p : float
        Dropout probability applied to the output tensor

    Attributes
    ---------
    scale : float
        scaling factor to address vanishing gradients. default: 1/sqrt{head_dim}
    
    qkv : nn.Linear
        linear projection for qkv

    proj : nn.Linear
        See attention paper: linear projection on output of transformer blocks

    attn_drop, proj_drop : nn.Dropout
        Dropout layers
    
    """
    def __init__(self, dim, num_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_size = dim // num_heads
        self.scale = self.head_size ** -0.5

        # instead of 3 separate linear mappings, combine them into 1
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Shape '(batch_size, num_patches + 1, dim)'

        Returns
        -------
        torch.Tensor
            Shape '(batch_size, num_patches + 1, embed_dim)'
        
        """
        batch_size, num_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (batch_size, num_patches + 1, 3 * dim)
        qkv = qkv.view(
            batch_size, num_tokens, 3, self.num_heads, self.head_size
            ) # (batch_size, num_patches + 1, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) #(3, batch_size, num_heads, num_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        scaled_matmul = q @ k_t * self.scale # (batch_size, num_heads, num_patches + 1, num_patches + 1)
        attn = scaled_matmul.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v #(batch_size, num_heads, num_patches + 1, head_dim)
        weighted_avg = weighted_avg.view(1, 2) #(batch_size, num_patches + 1, num_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) #(batch_size, num_patches + 1, dim) dim --> concatted all heads

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x
    



#from lucidrains ViT implementaiton
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

class MLP(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
                nn.LayerNorm(embed_size),
                nn.Lienar(embed_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_size)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block
    
    Parameters
    ----------
    dim : int
        embedding dimension
    
    num_heads : int
        number of attention heads

    mlp_ratio : float
        determines hidden dimension size of MLP w.r.t 'dim'

    qkv_bias : bool
        If True, add bias to qkv projections

    p, attn_p : float
        dropout probability

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization

    attn : Attention
        Attention module

    mlp : MLP
        MLP odule    
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.atn = Attention(dim, num_heads, qkv_bias, attn_p=attn_p, proj_p=p)
        # 2 separate layer norms so they have their own parameters
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(embed_size=dim, hidden_dim=hidden_dim)

    def forward(self, x):
        """run forward pass.
        Parameters
        ----------
        x : torch.tensor
            shape '(batch_size, num_patches + 1, dim)
        
        """
        x = x + self.atn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransfomer(nn.Module):
    def __init__(
            self, 
            img_size=384, 
            patch_size=16, 
            in_channels=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
            ):
        super().__init__()

        self.patch_embed = PatchEmbeddings(
                                        image_size=img_size, 
                                        embed_size=embed_dim,
                                        in_channels=in_channels,
                                        patch_size=patch_size
                                        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            p=p,
            attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1) #(batch_size, 1, embed_dim)

        x = torch.cat((cls_token, x), dim=1) # (batch_size, 1 + num_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        cls_token_final = x[:, 0] # fetch CLS token
        x = self.mlp_head(cls_token_final)

        return x