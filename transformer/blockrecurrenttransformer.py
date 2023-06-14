

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer, RecurrentLayer

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer as BlockRecurrentTransformerLucidrains


class BlockRecurrentTransformer(nn.Module):
    """
    Block Recurrent Transformer with a recurrent attention layer sandwiched in between.

    Parameters:
    vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens.
    max_len (int): The maximum length of the sequences that the transformer can handle.
    n_layers (int): The number of transformer layers in the model.
    d_model (int): The dimension of the transformer model, which is the size of the output vectors.
    n_head (int): The number of attention heads in the model.
    p (float): The dropout probability.
    device (str): The device to run the model on. Defaults to "cuda" if available.
    """

    def __init__(self,
                 vocab_size: int,
                 max_len: int = 512,
                 n_layers: int = 4,
                 d_model: int = 512,
                 n_head: int = 8,
                 p: float = 0.1,
                 device: str = None):
        super(BlockRecurrentTransformer, self).__init__()
        self.d_model = d_model
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len
                                              )
        self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p
                                                     )
                                      for _ in range(n_layers // 2)])
        self.recurrent = RecurrentLayer(d_model=d_model,
                                        ffn_hidden=4 * d_model,
                                        n_head=n_head,
                                        p=p
                                        )
        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p
                                                     )
                                      for _ in range(n_layers - n_layers // 2)])

    def init_state(self, batch_size: int, state_len: int):
        """
        Initializes the state of the model.

        Parameters:
        batch_size (int): The number of samples in a batch.
        state_len (int): The length of the state.

        Returns:
        Tensor: A tensor of random numbers with size (batch_size, state_len, self.d_model).
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        return torch.randn(batch_size, state_len, self.d_model, device=self.device)

    def state_forward(self, ids, state):
        """
        Computes the next state of the model.

        Parameters:
        ids (Tensor): A tensor containing the input token IDs.
        state (Tensor): The current state of the model.

        Returns:
        Tensor: The next state of the model.
        """
        if ids is None or state is None:
            raise ValueError("IDs and state cannot be None.")
        x = self.embedding(ids)
        for layer in self.layers1:
            x = layer(x)
        _, state = self.recurrent(x, state)

        return state

    def forward(self, ids, state):
        if ids is None or state is None:
            raise ValueError("IDs and state cannot be None.")
        x = self.embedding(ids)
        for layer in self.layers1:
            x = layer(x)
        x, state = self.recurrent(x, state)
        for layer in self.layers2:
            x = layer(x)
        return x, state


class BlockRecurrentTransformerPrewritten(nn.Module):
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

        super(BlockRecurrentTransformerPrewritten, self).__init__()
        self.max_len = max_len
        self.device = device

        self.model = BlockRecurrentTransformerLucidrains(
            num_tokens=vocab_size,
            dim=d_model,
            depth=n_layers,
            heads=n_head,
            max_seq_len=max_len
        )

    def init_state(self):
        return torch.randint(0, 2000, (1, self.max_len))

    def state_forward(self, state):
        return state

    def forward(self, ids, state):
        return ids, state

