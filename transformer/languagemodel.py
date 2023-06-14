

import torch.nn as nn


class TransformerLM(nn.Module):
    """
    Any kind of transformer cls has parameters:
        vocab_size
        max_len
        n_layers
        d_model
        n_head
        p

    and functions:
        state_forward(x, state) -> state
        forward(x,state) -> x, state

    """

    def __init__(self,
                 vocab_size: int,
                 max_len: int = 512,
                 n_layers: int = 4,
                 d_model: int = 512,
                 n_head: int = 8,
                 p: float = 0.1,
                 device: str = None):

        super(TransformerLM, self).__init__()

        self.transformer = cls(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def init_state(self):
        return self.transformer.init_state()

    def state_forward(self, x, state):
        return self.transformer.state_forward(x, state)

    def forward(self, x, state):
        x, state = self.transformer(x, state)
        x = self.lm_head(x)

        return x, state

