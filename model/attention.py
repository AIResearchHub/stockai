
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Attention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mask=None):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        out, attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        Split tensor into number of head

        Parameters:
        tensor : [batch_size, length, d_model]

        Returns:
        tensor : [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        Parameters:
        tensor : [batch_size, head, length, d_tensor]
        Returns:
        tensor : [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class XLAttention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(XLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mem=None, mask=None):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]
        mems:  [batch_size, mem_length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        batch_size, length, d_model = q.shape

        if mem is not None:
            c = torch.concat([mem, kv], dim=1)
        else:
            c = kv

        # q [batch_size, length, d_model]
        # c [batch_size, length+mem_length, d_model]
        q, k, v = self.w_q(q), *self.w_kv(c).chunk(2, dim=-1)
        q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        # q  [batch_size, n_head, length, d_head]
        # k  [batch_size, n_head, length+mem_length, d_head]
        attn_score = torch.einsum('bhid,bojd->bhij', (q, k)) / math.sqrt(self.d_head)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -10000)

        attn_prob = F.softmax(attn_score, dim=-1)

        # attn_prob [batch_size, n_head, length, length+mem_length]
        # v         [batch_size, n_head, length+mem_length, d_head]
        out = (attn_prob @ v).transpose(1, 2).reshape(batch_size, length, d_model)
        out = self.w_concat(out)

        # out [batch_size, length, d_model]
        assert out.shape == (batch_size, length, d_model)

        return out

    def split(self, tensor):
        tensor = tensor.view(tensor.size(0), tensor.size(1), self.n_head, self.d_head)
        tensor = tensor.transpose(1, 2)

        return tensor


class RecurrentAttention(nn.Module):
    """
    Recurrent Attention module for Block Recurrent Transformer Recurrent Layer
    See https://arxiv.org/pdf/2203.07852.pdf (page 2)
    This attention computes 4 separate queries, 2 keys and 2 values
    from input and recurrent state respectively then
    performs self attention and cross attention

    Parameters:
    d_model (int): Dimension of model
    n_head (int): Number of attention heads

    """

    def __init__(self, d_model, n_head):
        super(RecurrentAttention, self).__init__()
        self.n_head = n_head

        # get q, k, v for x and state
        self.w_qx1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qx2 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs2 = nn.Linear(d_model, d_model, bias=False)

        self.w_kvx = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_kvs = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, qx, kvx, qs, kvs, mask=None):
        """
        Computes recurrent attention for block recurrent transformer

        Parameters:
        qx (Tensor[batch_size, length, d_model]): input query
        kx (Tensor[batch_size, length, d_model]): input key
        vx (Tensor[batch_size, length, d_model]): input value
        qs (Tensor[batch_size, length, d_model]): state query
        ks (Tensor[batch_size, length, d_model]): state key
        vs (Tensor[batch_size, length, d_model]): state value
        """
        # compute 4 distinct queries
        qx1, qs1, qx2, qs2 = self.w_qx1(qx), self.w_qs1(qs), self.w_qx2(qx), self.w_qs2(qs)
        qx1, qs1, qx2, qs2 = self.split(qx1), self.split(qs1), self.split(qx2), self.split(qs2)

        # compute shared keys and values
        kx, vx = self.w_kvx(kvx).chunk(2, dim=-1)
        kx, vx = kx.unsqueeze(1), vx.unsqueeze(1)

        ks, vs = self.w_kvs(kvs).chunk(2, dim=-1)
        ks, vs = ks.unsqueeze(1), vs.unsqueeze(1)

        # perform self attention and cross attention
        x, _ = F.scaled_dot_product_attention(qx1, kx, vx, attn_mask=mask)
        s, _ = F.scaled_dot_product_attention(qs1, ks, vs, attn_mask=mask)

        xs, _ = F.scaled_dot_product_attention(qx2, ks, vs, attn_mask=mask)
        sx, _ = F.scaled_dot_product_attention(qs2, kx, vx, attn_mask=mask)

        # concatenate and linear projection
        x_proj = self.concat(torch.concat((xs, x), dim=-1))
        s_proj = self.concat(torch.concat((sx, s), dim=-1))

        x_proj = self.x_proj(x_proj)
        s_proj = self.s_proj(s_proj)

        return x_proj, s_proj

    def split(self, tensor):
        """
        Split tensor into number of head

        Parameters:
        tensor : [batch_size, length, d_model]

        Returns:
        tensor : [batch_size, head, length, d_tensor]

        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        Parameters:
        tensor : [batch_size, head, length, d_tensor]

        Returns:
        tensor : [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
