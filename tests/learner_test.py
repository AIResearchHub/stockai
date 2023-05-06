

from agent import Learner

import os
os.chdir("../")


learner = Learner(buffer_size=4,
                  batch_size=2,
                  n_accumulate=1,
                  tickers=["AAPL", "AMZN"],
                  mock_data=True,
                  vocab_size=30522,
                  n_layers=2,
                  d_model=12,
                  n_head=8,
                  n_cos=64,
                  n_tau=64,
                  n_p=128,
                  state_len=1,
                  n_step=1,
                  burnin_len=3,
                  rollout_len=6
                  )

import torch
import numpy as np
import pandas as pd

# def train_step(self, allocs, ids, actions, rewards, bert_targets, states):
#     """
#     :param allocs:       [block_len+n_step, batch_size*n_accumulate, 1]
#     :param ids:          [block_len+n_step, batch_size*n_accumulate, 501]
#     :param actions:      [block_len+n_step, batch_size*n_accumulate, 1, 1]
#     :param rewards:      [block_len, batch_size*n_accumulate, 1]
#     :param bert_targets: [block_len+n_step, batch_size*n_accumulate, 1]
#     :param states:       [batch_size*n_accumulate, state_len, d_model]
#     """

for i in range(100):
    learner.train_step(allocs=torch.zeros((10, 2, 1)).cuda(),
                       ids=torch.zeros((10, 2, 1), dtype=torch.int32).cuda(),
                       actions=torch.zeros((10, 2, 1, 1)).cuda(),
                       rewards=torch.zeros((9, 2, 1)).cuda(),
                       bert_targets=torch.zeros((10, 2, 1), dtype=torch.int64).cuda(),
                       states=torch.zeros((2, 1, 12)).cuda()
                       )

    print(learner.get_action(alloc=0.5,
                             timestamp=pd.Timestamp("2020-01-01"),
                             tickers=["AAPL", "AMZN"],
                             state=np.zeros((1, 12))))

