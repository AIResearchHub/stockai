

import torch.multiprocessing as mp

from agent import ReplayBuffer
from utils import read_context

sample_queue = mp.Queue()
batch_queue = mp.Queue(8)
priority_queue = mp.Queue(8)

contexts = read_context(tickers=["APPL", "AMZN", "BAC"],
                        mock_data=True
                        )

buffer = ReplayBuffer(buffer_size=10,
                      batch_size=3,
                      block_len=2 + 4,
                      n_step=1,
                      contexts=contexts,
                      sample_queue=sample_queue,
                      batch_queue=batch_queue,
                      priority_queue=priority_queue
                      )
