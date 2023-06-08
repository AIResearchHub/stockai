

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Critic(nn.Module):

    def __init__(self,
                 cls,
                 vocab_size=30522,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 n_cos=64
                 ):
        super(Critic, self).__init__()

        # hyper-parameters
        self.d_model = d_model

        # alloc head, policy_head and join fc
        self.alloc_head = nn.Linear(1, d_model)
        self.policy_head = nn.Linear(1, d_model)
        self.merge1 = nn.Linear(d_model, d_model)
        self.merge2 = nn.Linear(d_model, d_model)

        # transformer
        self.transformer = cls(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        # separate two heads
        self.bert_head = nn.Linear(d_model, vocab_size)
        self.critic_head = nn.Linear(d_model, d_model)

        # iqn
        self.iqn = IQN(
            d_model=d_model,
            n_cos=n_cos
        )

        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, 1)

    def init_state(self, batch_size=1, device="cpu"):
        return self.transformer.init_state(batch_size=batch_size, device=device)

    def state_forward(self, ids, state):
        return self.transformer.state_forward(ids, state)

    def forward(self, x, p, state, n_tau):
        # x = allocs
        # b = ids
        # p = policies

        (x, b) = x
        batch_size = x.size(0)

        # compute alloc, ids, policy
        x = F.gelu(self.alloc_head(x))
        b, state = self.transformer(b, state)
        p = F.gelu(self.policy_head(p))

        # separate transformer output into two heads
        bert = F.log_softmax(self.bert_head(b))
        b = self.critic_head(b.mean(dim=1))

        # join alloc and ids
        x = self.merge1(x * b)
        x = x.view(batch_size, self.d_model)

        # join alloc, ids, policy
        assert x.shape == (batch_size, self.d_model)
        assert p.shape == (batch_size, self.d_model)

        x = self.merge2(x * p)
        x = x.unsqueeze(1)

        # pass in iqn
        x, taus = self.iqn(x, n_tau=n_tau)
        return (x, bert), taus, state


class Policy(nn.Module):

    def __init__(self,
                 cls,
                 vocab_size=30522,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):
        super(Policy, self).__init__()

        # hyper-parameters
        self.d_model = d_model

        # alloc head, policy_head and join fc
        self.alloc_head = nn.Linear(1, d_model)

        # transformer
        self.transformer = cls(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        # out head
        self.linear = nn.Linear(d_model, d_model)
        self.merge = nn.Linear(d_model, d_model)
        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, 1)

    def init_state(self, batch_size=1, device="cpu"):
        return self.transformer.init_state(batch_size=batch_size, device=device)

    def state_forward(self, ids, state):
        return self.transformer.state_forward(ids, state)

    def forward(self, x, state):
        x, b = x
        # compute alloc, ids, policy
        x = F.gelu(self.alloc_head(x))
        b, state = self.transformer(b, state)

        # separate transformer output into two heads
        b = self.linear(b.mean(dim=1))

        # join alloc and ids
        x = F.gelu(self.merge(x * b))
        x = F.gelu(self.out1(x))
        x = F.tanh(self.out2(x))

        # pass in iqn
        return x, state


class IQN(nn.Module):
    """
    Implicit Quantile Network
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
        :param x:     Tensor[batch_size, 1, d_model]
        :param n_tau: int
        :return:      Tensor[batch_size, n_tau]
                      Tensor[batch_size, n_tau, 1]
        """
        batch_size = x.size(0)
        assert x.shape == (batch_size, 1, self.d_model)

        cos, taus = self.calc_cos(batch_size, n_tau=n_tau)

        cos = cos.view(batch_size, n_tau, self.d_model)
        taus = taus.view(batch_size, n_tau)

        x = (x*cos).view(batch_size*n_tau, self.d_model)
        x = self.gelu(self.linear(x))
        x = self.out(x)
        x = x.view(batch_size, n_tau)

        return x, taus
