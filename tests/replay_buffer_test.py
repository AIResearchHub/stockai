

import torch.multiprocessing as mp
import numpy as np

from agent import ReplayBuffer
from agent.replay_buffer import Episode
from utils import read_context

import os
os.chdir("../")

sample_queue = mp.Queue()
batch_queue = mp.Queue(8)
priority_queue = mp.Queue(8)

contexts = read_context(tickers=["AAPL", "AMZN"],
                        mock_data=True
                        )

buffer = ReplayBuffer(buffer_size=10,
                      batch_size=3,
                      block_len=2 + 4,
                      d_model=12,
                      state_len=1,
                      n_step=1,
                      gamma=0.99,
                      contexts=contexts,
                      sample_queue=sample_queue,
                      batch_queue=batch_queue,
                      priority_queue=priority_queue
                      )


buffer.start_threads()

import pandas as pd

episode = Episode(tickers=np.array(["AAPL", "AMZN"]),
                  allocs=np.arange(30,),
                  timestamps=np.full(40, fill_value=pd.Timestamp("2019-01-01")),
                  actions=np.arange(30,),
                  rewards=np.arange(30,),
                  states=np.zeros((30, 1, 12)),
                  length=30,
                  total_reward=10,
                  total_time=100)

sample_queue.put(episode)

episode = Episode(tickers=np.array(["AMZN", "AAPL"]),
                  allocs=np.arange(40,),
                  timestamps=np.full(40, fill_value=pd.Timestamp("2020-01-01")),
                  actions=np.arange(40,),
                  rewards=np.arange(40,),
                  states=np.zeros((40, 1, 12)),
                  length=40,
                  total_reward=5,
                  total_time=100)

sample_queue.put(episode)

import time
while True:
    time.sleep(1)
    priority_queue.put(([[0, 0], [1, 0], [0, 0]], np.ones((3, 7, 1, 12)), 0, 0, .4))

    print(batch_queue.qsize())
    print(batch_queue.get())

