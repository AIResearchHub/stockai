

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer


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
                 vocab_size: int,
                 max_len: int = 512,
                 n_layers: int = 4,
                 d_model: int = 512,
                 n_head: int = 8,
                 p: float = 0.1,
                 device: str = None):
        super(Transformer, self).__init__()

        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                    ffn_hidden=4 * d_model,
                                                    n_head=n_head,
                                                    p=p)
                                     for _ in range(n_layers)])

    def init_state(self, batch_size: int = 1, device: str = "cpu"):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        return torch.zeros(1, batch_size, 1, 1, device=device)

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
        if ids is None or state is None:
            raise ValueError("IDs and state cannot be None.")
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)

        return x, state
