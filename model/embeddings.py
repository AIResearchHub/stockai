

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embedding
    """

    def __init__(self, d_model, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model),
                                     requires_grad=True)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :]


class TokenEmbedding(nn.Module):
    """
    Token Embedding
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        """
        :param  [batch_size, length]
        :return [batch_size, length, dim]
        """
        token_emb = self.emb(ids)
        return token_emb


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding
    """

    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model, max_len)

    def forward(self, x):
        """
        :param  [batch_size, length]
        :return [batch_size, length, dim]
        """
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        return token_emb + pos_emb
