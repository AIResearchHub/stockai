

import torch
import torch.nn as nn

from .embeddings import TransformerEmbedding
from .layers import AttentionLayer, RecurrentAttentionLayer

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer \
    as BlockRecurrentTransformerLucidrains


class Transformer(nn.Module):
    """
    A standard Transformer module
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

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                    ffn_hidden=4 * d_model,
                                                    n_head=n_head,
                                                    p=p)
                                    for _ in range(n_layers)])

    def state_forward(self, ids, state):
        return state

    def forward(self, ids, state):
        """
        :param ids: torch.Tensor [batch_size, length]
        :return: torch.Tensor [batch_size, length, d_model]
        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)

        return x, state


class BlockRecurrentTransformer(nn.Module):
    """
    A simplified implementation of BlockRecurrentTransformer see
    https://arxiv.org/pdf/2203.07852.pdf
    without the transformer xl memory and cache
    """

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

        # self.load_embeddings()

    def load_embeddings(self):
        """
        uses bert-base-uncased embeddings weights for token and position embedding
        """
        tok_dict = torch.load("saved/word_embeddings")
        pos_dict = torch.load("saved/position_embeddings")

        tok_dict["emb.weight"] = tok_dict["weight"]
        pos_dict["encoding"] = pos_dict["weight"]

        del tok_dict["weight"]
        del pos_dict["weight"]

        self.embedding.tok_emb.load_state_dict(tok_dict)
        self.embedding.pos_emb.load_state_dict(pos_dict)
        self.recurrent.state_ids.load_state_dict(pos_dict)

    def load_pretrained_bert(self):
        """
        only compatible if d_model = 768 and n_head = 8

        TODO: not done, transformer layer architecture not compatible
        """
        assert self.d_model == 768
        assert self.n_head == 8

        from pytorch_pretrained_bert import BertModel
        bert = BertModel.from_pretrained('bert-base-uncased')

        for layer in [*self.layer1, *self.layer2]:
            pass

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model).cuda()

    def state_forward(self, ids, state):
        """
        :param ids:   Tensor[batch_size, length]
        :param state: Tensor[batch_size, state_len, d_model]
        :return:      Tensor[batch_size, state_len, d_model]
        """

        x = self.embedding(ids)
        # x, _ = self.statein(x, state.detach())

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
        # x, _ = self.statein(x, state.detach())

        for layer in self.layers1:
            x = layer(x)
        x, state = self.recurrent(x, state)
        for layer in self.layers2:
            x = layer(x)

        return x, state


class BlockBERTlucidrains(nn.Module):
    """
    lucidrains' block recurrent transformer taken from
    https://github.com/lucidrains/block-recurrent-transformer-pytorch
    many features are removed for simplicity
    """

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):
        super(BlockBERTlucidrains, self).__init__()

        self.transformer = BlockRecurrentTransformerLucidrains(
            num_tokens=vocab_size,
            dim=d_model,
            depth=n_layers,
            heads=n_head,
            num_state_vectors=1,
            max_seq_len=512
        )

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model).cuda()

    def state_forward(self, ids, state):
        _, state = self.transformer.forward(ids, states=[state])
        return state[0]

    def forward(self, ids, state):
        x, state = self.transformer.forward(ids, states=[state])
        state = state[0]

        return x, state

