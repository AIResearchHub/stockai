

import torch
import torch.nn as nn

from .embeddings import TransformerEmbedding
from .layers import AttentionLayer, RecurrentAttentionLayer


class BlockRecurrentTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):

        super(BlockRecurrentTransformer, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.statein = RecurrentAttentionLayer(d_model=d_model,
                                               ffn_hidden=4 * d_model,
                                               n_head=n_head,
                                               p=p)
        self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                      for _ in range(n_layers//2)])
        self.recurrent = RecurrentAttentionLayer(d_model=d_model,
                                                 ffn_hidden=4 * d_model,
                                                 n_head=n_head,
                                                 p=p)
        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                      for _ in range(n_layers//2)])

    def init_state(self, batch_size, state_len):
        return torch.zeros(
            batch_size,
            state_len,
            self.d_model
        ).cuda()

    def state_forward(self, ids, state):
        """
        :param ids:   Tensor[batch_size, length]
        :param state: Tensor[batch_size, state_len, d_model]
        :return:      Tensor[batch_size, state_len, d_model]
        """

        x = self.embedding(ids)
        x, _ = self.statein(x, state.detach())

        for layer in self.layers1:
            x = layer(x)
        _, state = self.recurrent(x, state)

        return state

    def forward(self, ids, state):
        """
        :param ids:   Tensor[batch_size, length]
        :param state: Tensor[batch_size, state_len, d_model]
        :return:      Tensor[batch_size, 1]
                      Tensor[batch_size, state_len, d_model]
        """

        x = self.embedding(ids)
        x, _ = self.statein(x, state.detach())

        for layer in self.layers1:
            x = layer(x)
        x, state = self.recurrent(x, state)
        for layer in self.layers2:
            x = layer(x)

        return x, state
