import torch
import torch.nn as nn

from .embeddings import TransformerEmbedding
from .layers import AttentionLayer, LongformerAttentionLayer, XLAttentionLayer, RecurrentAttentionLayer

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer as BlockRecurrentTransformerLucidrains


class Transformer(nn.Module):
    """
    A standard Transformer module that outputs the unprocessed
    output of the last transformer layer

    Parameters:
    vocab_size (int): Vocabulary size
    max_len (int): Max length
    n_layers (int): Number of layers
    d_model (int): Dimension of transformer
    n_head (int): Number of attention heads
    p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):
        super(Transformer, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
                                     for _ in range(n_layers)])

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model, device=self.device)

    def state_forward(self, ids, state):
        """Returns next recurrent state, since standard transformer just return original state"""
        return state

    def forward(self, ids, state):
        """
        Computes transformer output

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)

        return x, state


class TransformerXL(nn.Module):
    """
    A standard Transformer module that outputs the unprocessed
    output of the last transformer layer

    Parameters:
    vocab_size (int): Vocabulary size
    max_len (int): Max length
    n_layers (int): Number of layers
    d_model (int): Dimension of transformer
    n_head (int): Number of attention heads
    p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda"
                 ):

        super(TransformerXL, self).__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.layers = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                      ffn_hidden=4 * d_model,
                                                      n_head=n_head,
                                                      p=p)
                                    for _ in range(n_layers)])

    def init_state(self, batch_size=1, device="cpu"):
        return torch.zeros(self.n_layers, batch_size, self.max_len, self.d_model, device=device)

    def state_forward(self, ids, state):
        """Returns next recurrent state, since standard transformer just return original state"""
        x = self.embedding(ids)

        next_state = []
        for layer, s in zip(self.layers, state):
            next_state.append(x.detach())
            x = layer(x, s)

        next_state = torch.stack(next_state)
        return next_state

    def forward(self, ids, state):
        """
        Computes transformer xl output
        Layer takes in (length, batch_size, d_model) so transpose before and after layers

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        x = self.embedding(ids)

        next_state = []
        for layer, s in zip(self.layers, state):
            next_state.append(x.detach())
            x = layer(x, s)

        next_state = torch.stack(next_state)
        return x, next_state


class Longformer(nn.Module):
    """
    A standard Longformer module that outputs the unprocessed
    output of the last transformer layer

    Parameters:
    vocab_size (int): Vocabulary size
    max_len (int): Max length
    n_layers (int): Number of layers
    d_model (int): Dimension of transformer
    n_head (int): Number of attention heads
    p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):
        super(Longformer, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([LongformerAttentionLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
                                     for _ in range(n_layers)])

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model, device=self.device)

    def state_forward(self, ids, state):
        """Returns next recurrent state, since standard transformer just return original state"""
        return state

    def forward(self, ids, state):
        """
        Computes transformer output

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)

        return x, state


class BlockRecurrentTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=512, n_layers=4, d_model=512, n_head=8, p=0.1, device="cuda"):
        super(BlockRecurrentTransformer, self).__init__()
        self.device = torch.device(device)
        self.d_model = d_model

        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
                                      for _ in range(n_layers // 2)])
        self.recurrent = RecurrentAttentionLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
                                      for _ in range(n_layers - n_layers // 2)])

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model, device=self.device)

    def state_forward(self, ids, state):
        x = self.embedding(ids)
        for layer in self.layers1:
            x = layer(x)
        _, state = self.recurrent(x, state)

        return state

    def forward(self, ids, state):
        x = self.embedding(ids)
        for layer in self.layers1:
            x = layer(x)
        x, state = self.recurrent(x, state)
        for layer in self.layers2:
            x = layer(x)
        return x, state


class BlockBERTlucidrains(nn.Module):
    def __init__(self, vocab_size, n_layers=4, d_model=128, n_head=8, p=0.1, device="cuda"):
        super(BlockBERTlucidrains, self).__init__()
        self.d_model = d_model
        self.device = torch.device(device)

        self.transformer = BlockRecurrentTransformerLucidrains(num_tokens=vocab_size, dim=d_model, depth=n_layers, heads=n_head, num_state_vectors=1, max_seq_len=512)

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model, device=self.device)

    def forward(self, ids, state):
        x, state = self.transformer.forward(ids, states=[state])
        state = state[0]
        return x, state
