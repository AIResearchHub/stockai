

import torch
import torch.nn as nn

import numpy as np

from .transformer import Transformer, BlockRecurrentTransformer, BlockBERTlucidrains


class Model(nn.Module):
    """
    A BlockRecurrentTransformer module followed by a IQN Module
    The Transformer does all the heavy work while the
    IQN uses the output embedding to create a q value distribution

    Parameters:
    vocab_size (int): vocabulary size of the transformer
    max_len (int): max length of the transformer
    n_layers (int): number of layers in transformer
    n_head (int): number of attention heads in transformer
    p (float): dropout probability in transformer
    n_cos (int): number of cosine samples for each tau in IQN
    """

    def __init__(self,
                 vocab_size=30522,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 n_cos=64
                 ):
        super(Model, self).__init__()

        # hyper-parameters
        self.d_model = d_model

        # alloc head, policy_head and join fc
        self.alloc_head = nn.Linear(1, d_model)
        self.policy_head = nn.Linear(1, d_model)
        self.merge1 = nn.Linear(d_model, d_model)
        self.merge2 = nn.Linear(d_model, d_model)

        # transformer
        self.transformer = Transformer(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )
        # self.transformer = BlockRecurrentTransformer(
        #     vocab_size=vocab_size,
        #     max_len=max_len,
        #     n_layers=n_layers,
        #     d_model=d_model,
        #     n_head=n_head,
        #     p=p
        # )
        # self.transformer = BlockBERTlucidrains(
        #     vocab_size=vocab_size,
        #     n_layers=n_layers,
        #     d_model=d_model,
        #     n_head=n_head,
        #     p=p
        # )

        # separate two heads
        self.bert_head = nn.Linear(d_model, vocab_size)
        self.critic_head = nn.Linear(d_model, d_model)

        # self.bert_head = nn.Sequential(
        #     nn.Linear(d_model, vocab_size),
        #     nn.LogSoftmax(dim=-1)
        # )
        # self.critic_head = nn.Linear(d_model, d_model)

        # iqn
        self.iqn = IQN(
            d_model=d_model,
            n_cos=n_cos
        )

        # non-linearity
        self.gelu = nn.GELU()
        self.softmax = nn.LogSoftmax(dim=-1)

        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, 1)

    def init_state(self, batch_size=1, state_len=1):
        """
        Return initialized recurrent state

        Returns:
        init_state (Tensor[batch_size, state_len, d_model]): either zeros, random normal or learned parameters
        """
        return self.transformer.init_state(batch_size, state_len)

    def state_forward(self, ids, state):
        """
        Returns the next recurrent state without computing output,
        saves compute by not passing through remaining transformer layers
        and IQN

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        state (Tensor[batch_size, state_len, d_model]): next recurrent state

        """
        return self.transformer.state_forward(ids, state)

    def forward(self, xp, state, n_tau):
        """

        Parameters:
        xp (Tensor[batch_size, length], Tensor[batch_size, n_p, 1]): tokens and policy
        state (Tensor[batch_size, state_len, d_model]): recurrent state
        n_tau (int): number of tau samples to approximate distribution

        Returns:
        (x, bert) (Tensor[batch_size, num_p, n_tau], Tensor[batch_size, max_len, vocab_size]): critic output and bert output
        taus (Tensor[batch_size, n]): tau values for the distribution, representing percentile in distribution
        state (Tensor[batch_size, state_len, d_model]): next recurrent state

        """

        # x = allocs
        # b = ids
        # p = policies

        (x, b), p = xp
        batch_size = p.size(0)
        n_p = p.size(1)

        # compute alloc, ids, policy
        x = self.gelu(self.alloc_head(x))
        b, state = self.transformer(b, state)
        p = self.gelu(self.policy_head(p))

        # separate transformer output into two heads
        bert = self.softmax(self.bert_head(b))
        b = self.critic_head(b.mean(dim=1))

        # join alloc and ids
        x = self.merge1(x * b)
        x = x.view(batch_size, 1, self.d_model)

        # join alloc, ids, policy
        assert x.shape == (batch_size, 1, self.d_model)
        assert p.shape == (batch_size, n_p, self.d_model)

        x = self.merge2(x * p)
        x = x.unsqueeze(2)

        # pass in iqn
        x, taus = self.iqn(x, n_tau=n_tau)
        return (x, bert), taus, state


class IQN(nn.Module):
    """
    Implicit Quantile Network see https://arxiv.org/pdf/1806.06923.pdf
    This model outputs a q value distribution instead of a scalar
    And is trained using quantile loss to predict the value at each
    percentile of the distribution, designed to approximate distribution
    from output representation of a backbone model

    Parameters:
    d_model (int): The dimension of the model
    n_cos (int): Number of cosine samples for each tau
    """

    def __init__(self,
                 d_model,
                 n_cos=64
                 ):
        super(IQN, self).__init__()

        self.cos_embedding = nn.Linear(n_cos, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)

        self.n_cos = n_cos
        self.d_model = d_model

        self.pis = torch.FloatTensor([np.pi*i for i in range(1, n_cos+1)]).view(1, 1, n_cos).cuda()

        self.gelu = nn.GELU()

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the co-sin values depending on the number of tau samples

        Parameters:
        batch_size (int): batch size

        Returns:
        cos (Tensor[batch_size, n_tau, d_model]): cosine values to be bitwise multiplied with output representation
        taus (Tensor[batch_size, n_tau]): tau values associated with each cosine representation
        """
        assert torch.equal(self.pis,
                           torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1, 1, self.n_cos).cuda())

        # (batch_size, n_tau, 1)
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).cuda()
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos)

        cos = cos.view(batch_size * n_tau, self.n_cos)
        cos = self.gelu(self.cos_embedding(cos))
        cos = cos.view(batch_size, n_tau, self.d_model)

        return cos, taus

    def forward(self, x, n_tau):
        """
        Additional axis to incorporate multiple policy values

        Parameters:
        x (Tensor[batch_size, n_p, 1, d_model]): output representation of backbone model
        n_tau (int): Number of tau samples

        Returns:
        x (Tensor[batch_size, n_p, n_tau]): output distribution
        taus (Tensor[batch_size, n_p, n_tau]): tau samples associated with each distribution

        """
        batch_size = x.size(0)
        n_p = x.size(1)

        assert x.shape == (batch_size, n_p, 1, self.d_model)

        cos, taus = self.calc_cos(batch_size*n_p, n_tau=n_tau)

        cos = cos.view(batch_size, n_p, n_tau, self.d_model)
        taus = taus.view(batch_size, n_p, n_tau)

        x = (x*cos).view(batch_size*n_p*n_tau, self.d_model)
        x = self.gelu(self.linear(x))
        x = self.out(x)
        x = x.view(batch_size, n_p, n_tau)

        return x, taus
