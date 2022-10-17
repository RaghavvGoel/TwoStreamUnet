import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# torch.manual_seed(42)

class PreNorm(nn.Module):
    def __init__(self, hidden_dim, fn) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fn = fn 
    
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # self.attend = nn.Softmax(dim = -1)
        # self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, z, x):
        # x is [x_tilde, K_prev]
        # z is the measurement error
        q = self.to_q(z)
        k = self.to_k(x)
        v = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = nn.functional.softmax(dots, dim=-1)
        # attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # self.layers.append(nn.ModuleList([
            #     PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            #     PreNorm(dim, FeedForward(dim, dim, mlp_dim, dropout = dropout))
            # ]))
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, dim, mlp_dim, dropout=dropout),
                nn.LayerNorm(dim)
            ]))
    def forward(self, z, x):
        # z_x_K : concatenated along batch dim
        # Query => x[:B], Key, Value => x[B:]
        for attn, _, ff, _ in self.layers:
            z = attn(z, x) # + z
            z = ff(z) #+ z
        return z