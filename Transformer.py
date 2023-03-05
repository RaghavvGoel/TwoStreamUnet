import torch
import torch.nn as nn
from torch import Tensor
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
        self.to_q = nn.Linear(dim, inner_dim, bias = False) # we have subsumed this in encoder, so can be removed
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, z, x, bilateral_flag=False, bilateral_feat=None):
        #
        q = self.to_q(z)
        k = self.to_k(x)
        v = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if bilateral_flag:
            # print("dots shape {}, bilateral_feat shape {}".format(dots.shape, torch.stack([bilateral_feat]*self.heads, dim=1).shape))
            dots = dots + torch.stack([bilateral_feat]*self.heads, dim=1)
        attn = nn.functional.softmax(dots, dim=-1)
        # attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # n is number of nodes/elements
        # out = torch.mean(out, dim = 1) #* mean along head
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=128, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # self.layers.append(nn.ModuleList([
            #     PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            #     PreNorm(dim, FeedForward(dim, dim, mlp_dim, dropout = dropout))
            # ]))
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head = dim_head, dropout = dropout),
                nn.LayerNorm(dim)
                # FeedForward(dim, dim, mlp_dim, dropout=dropout),
                # nn.LayerNorm(dim)
            ]))
        #! add position embedding 
        # self.pos_embbeding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        
    def forward(self, z, x):
        # z_x_K : concatenated along batch dim
        # Query => x[:B], Key, Value => x[B:]
        # for attn, _, ff, _ in self.layers:
        #     z = attn(z, x)  + z
        #     z = ff(z) + z
        for attn, _, in self.layers:
            # take mean over heads
            z = attn(z, x)  + z

        return z

"""
Adapted from https://github.com/lukemelas/simple-bert
"""

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer_v2(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x        