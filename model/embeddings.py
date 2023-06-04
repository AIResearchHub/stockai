import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding from original transformer paper.

    Parameters:
    d_model (int): dimension of model
    max_len (int): max length of transformer
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)  # Keep this on CPU
        self.encoding.requires_grad = False  # No need for gradient

        pos = torch.arange(0, max_len).unsqueeze(dim=1).float()
        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """Obtain positional encoding according to input size"""
        self.encoding = self.encoding.to(x.device)  # Move to the same device as input
        return self.encoding[:x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embedding

    Parameters:
    d_model (int): Dimension of model
    max_len (int): Max length of transformer
    """

    def __init__(self, d_model, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model), requires_grad=True)

    def forward(self, x):
        """Return learned positional encoding according to input shape"""
        return self.encoding[:x.size(1), :]


class TokenEmbedding(nn.Module):
    """
    Token Embedding for transformer
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        """
        Parameters:
        ids : [batch_size, length]
        token_emb : [batch_size, length, dim]
        """
        return self.emb(ids)


class TransformerEmbedding(nn.Module):
    """
    This class manages the complete transformer embeddings consisting of both positional and token embeddings.
    """

    def __init__(self, vocab_size, d_model, max_len, positional_encoding='sinusoid'):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        if positional_encoding == 'sinusoid':
            self.pos_emb = PositionalEncoding(d_model, max_len)
        elif positional_encoding == 'learned':
            self.pos_emb = LearnedPositionalEncoding(d_model, max_len)
        else:
            raise ValueError(f"positional_encoding should be either 'sinusoid' or 'learned', but got {positional_encoding}")

    def forward(self, x):
        """
        Returns complete transformer embedding for transformer layers

        Parameters:
        x : [batch_size, length]

        Returns:
        token_emb + pos_emb : [batch_size, length, dim]
        """
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        return token_emb + pos_emb
