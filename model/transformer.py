import torch
import torch.nn as nn

from .embeddings import TransformerEmbedding
from .layers import AttentionLayer, RecurrentAttentionLayer

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer as BlockRecurrentTransformerLucidrains


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len=512, n_layers=4, d_model=512, n_head=8, p=0.1):
        super(Transformer, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
                                     for _ in range(n_layers)])

    def forward(self, ids):
        """
        :param ids: torch.Tensor [batch_size, length]
        :return: torch.Tensor [batch_size, length, d_model]
        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)

        return x


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

    def forward(self, ids, state):
        """
        :param ids:   Tensor[batch_size, length]
        :param state: Tensor[batch_size, state_len, d_model]
        :return:      Tensor[batch_size, 1]
                      Tensor[batch_size, state_len, d_model]
        """
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
