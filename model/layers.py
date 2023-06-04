import torch
import torch.nn as nn

from .attention import Attention, RecurrentAttention
from .embeddings import LearnedPositionalEncoding

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LongformerSelfAttention
# from transformers import BigBirdSelfAttention


class AxialAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head *  heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, h, w, _, heads = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = map(lambda t: rearrange(t, 'b h w (heads d) -> b heads h w d', heads = heads), qkv.chunk(3, dim = -1))

        dots = einsum('b h i j d, b h x y d -> b h i j x y', q, k) * (1. / sqrt(k.shape[-1]))
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j x y, b h x y d -> b h i j d', attn, v)
        out = rearrange(out, 'b heads h w d -> b h w (heads d)')
        return self.to_out(out)


class AxialAttentionLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AxialAttentionLayer, self).__init__()
        self.attention = AxialAttention(d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        _x = x
        x = self.attention(x)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class BigBirdAttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes BigBird self-attention,
    normalization, dropout, and a feed-forward network.

    Parameters:
    d_model (int): The dimension of the model.
    ffn_hidden (int): The size of the hidden layer in the feed-forward network.
    n_head (int): The number of attention heads.
    p (float): The probability of dropout.
    """

    def __init__(self, d_model, ffn_hidden, n_head, p, block_size, num_random_blocks):
        super(BigBirdAttentionLayer, self).__init__()
        self.config = BigBirdConfig.from_pretrained('google/bigbird-roberta-base', attention_type="block_sparse",
                                                    block_size=block_size, num_random_blocks=num_random_blocks)
        self.attention = BigBirdSelfAttention(self.config)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """
        Forward pass of the BigBirdAttentionLayer.

        Parameters:
        x (Tensor): Input tensor.
        src_mask (Tensor, optional): Source mask tensor.

        Returns:
        Tensor: Output tensor after passing through the layer.
        """
        _x = x
        x = self.attention(input_ids=x, attention_mask=src_mask)[0]

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class LongformerAttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes Longformer self-attention,
    normalization, dropout, and a feed-forward network.

    Parameters:
    d_model (int): The dimension of the model.
    ffn_hidden (int): The size of the hidden layer in the feed-forward network.
    n_head (int): The number of attention heads.
    p (float): The probability of dropout.
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(LongformerAttentionLayer, self).__init__()
        self.attention = LongformerSelfAttention(config=self.config)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """
        Forward pass of the LongformerAttentionLayer.

        Parameters:
        x (Tensor): Input tensor.
        src_mask (Tensor, optional): Source mask tensor.

        Returns:
        Tensor: Output tensor after passing through the layer.
        """
        _x = x
        x = self.attention(input_ids=x, attention_mask=src_mask)[0]

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class AttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes self-attention,
    normalization, dropout, and a feed-forward network.

    Parameters:
    d_model (int): The dimension of the model.
    ffn_hidden (int): The size of the hidden layer in the feed-forward network.
    n_head (int): The number of attention heads.
    p (float): The probability of dropout.
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AttentionLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """
        Forward pass of the AttentionLayer.

        Parameters:
        x (Tensor): Input tensor.
        src_mask (Tensor, optional): Source mask tensor.

        Returns:
        Tensor: Output tensor after passing through the layer.
        """
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class RecurrentAttentionLayer(nn.Module):
    """
    A recurrent transformer layer from block-recurrent transformer
    """

    def __init__(self, d_model, ffn_hidden, n_head, p, max_len=512):
        super(RecurrentAttentionLayer, self).__init__()

        # learned ids
        self.state_norm = nn.LayerNorm(d_model)
        self.state_ids = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)

        # attention
        self.attention = RecurrentAttention(d_model=d_model, n_head=n_head)

        # forget gates
        self.proj_gate = FixedGate(d_model)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

        # feed forward model
        self.ffn = FeedForward(d_model, inner_dim=ffn_hidden)

    def forward(self, x, s, x_mask=None, s_mask=None):
        _x = x
        _s = s

        s = self.state_norm(s) + self.state_ids(s)

        x_proj, s_proj = self.attention(qx=x, kx=x, vx=x, qs=s, ks=s, vs=s)

        # finish computing out
        x_residual = x_proj + _x
        out = self.ffn(x_residual) + x_residual

        # fixed simple gate
        next_s = self.proj_gate(s_proj, _s)

        return out, next_s


class FeedForward(nn.Module):
    """
    Sequential(
        Linear(dim, inner_dim)
        GELU()
        Linear(inner_dim, dim)
    )
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)


class FixedGate(nn.Module):
    """
    Fixed Gate for block-recurrent transformer
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.bias = nn.Parameter(torch.randn(dim), requires_grad=True)

    def forward(self, x, state):
        z = self.proj(x)
        g = torch.sigmoid(self.bias)
        return torch.mul(state, g) + torch.mul(z, 1 - g)
