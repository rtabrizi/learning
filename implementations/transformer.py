import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
embed_size = 384
context_len = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_heads = 6
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False) # y=xW^T + b --> W^T = (input_size, output_size)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        # for autocomplete, we need a mask so that the transformer doens't attend to future tokens (autoregressive)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, context_len, embed_size = x.shape
        k = self.key(x) # (batch_size, context_len, embed_size) @ (embed_size, head_size) --> (batch_size, context_len, head_size)
        q = self.query(x) # (batch_size, context_len, head_size)

        # need to compute the affinities, scaled attention
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (batch_size, context_len, num_heads) @ (batch_size, num_heads, context_len) 
                                                              # -> (batch_size, context_len, context_len)
        weights = weights.masked_fill(self.tril[:context_len, :context_len] ==0, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (batch_size, context_len, context_len)
        weights = self.dropout(weights)
        # now need to do weighted aggregation
        v = self.value(x) #(batch_size, context_len, head_size)
        out = weights @ v # (batch_size, context_len, context_len) @ (batch_size, context_len, num_heads) --> (batch_size, context_len, num_heads)
        return out

class MultiHeadSelfAttention(nn.Module):    
    def __init__(self, embed_size, num_heads):
        assert embed_size % num_heads == 0, "num heads not compatible "
        super().__init__()
        self.head_size = embed_size // num_heads
        self.heads = nn.ModuleList([SelfAttention(self.head_size) for _ in range(num_heads)])
        
        self.proj = nn.Linear(self.head_size * num_heads, embed_size)
        # same as embed_size so we can do skip connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #(batch_size, context_len, embed_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    """ A complete Transformer _decoder_ block"""

    def __init__(self, embed_size, num_heads):
        super().__init__()

        head_size = embed_size // num_heads
        self.sa = MultiHeadSelfAttention(head_size, num_heads)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.layer_norm(x))
        x = x + self.ffwd(self.layer_norm(x))
        return x