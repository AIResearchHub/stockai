import torch
import threading
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List

from .logger import Logger
from utils import get_context, mask_ids


@dataclass
class Episode:
    tickers: np.array
    allocs: np.array
    timestamps: np.array
    actions: np.array
    rewards: np.array
    states: np.array
    length: int
    total_reward: float
    total_time: float


@dataclass
class Block:
    allocs: torch.tensor
    ids: torch.tensor
    actions: torch.tensor
    rewards: torch.tensor
    bert_targets: torch.tensor
    states: torch.tensor
    idxs: List[List[int]]


class ReplayBuffer:

    def __init__(self,
                 buffer_size,
                 batch_size,
                 block_len,
                 d_model,
                 state_len,
                 n_step,
                 gamma,
                 contexts,
                 sample_queue,
                 batch_queue,
                 priority_queue
                 ):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.block_len = block_len

        self.d_model = d_model
        self.state_len = state_len
        self.n_step = n_step
        self.gamma = gamma

        self.contexts = contexts

        self.lock = threading.Lock()

        self.sample_queue = sample_queue
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue

        self.buffer = np.empty(shape=(buffer_size,), dtype=object)

        self.logger = Logger()

        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def start_threads(self):
        thread = threading.Thread(target=self.add_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.prepare_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.update_data, daemon=True)
        thread.start()

        thread = threading.Thread(target=self.log_data, daemon=True)
        thread.start()

    def add_data(self):
        """
        asynchronously add episodes to buffer
        """
        while True:
            time.sleep(0.1)

            if not self.sample_queue.empty():
                data = self.sample_queue.get_nowait()
                self.add(data)

    def prepare_data(self):
        """
        asynchronously add batches to batch_queue
        """
        while True:
            time.sleep(0.1)

            if not self.batch_queue.full() and self.size != 0:
                data = self.sample_batch()
                self.batch_queue.put(data)

    def update_data(self):
        """
        asynchronously update states inside buffer
        """
        while True:
            time.sleep(0.1)

            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)

    def log_data(self):
        """
        asynchronously prints out logs and write into file
        """
        while True:
            time.sleep(10)

            self.log()

    def add(self, episode):

        with self.lock:

            # add to buffer
            self.buffer[self.ptr] = episode

            # increment size
            self.size += 1
            self.size = min(self.size, self.buffer_size)

            # increment pointer
            self.ptr += 1
            self.ptr = self.ptr % self.buffer_size

            # log
            self.logger.total_frames += episode.length
            self.logger.reward = episode.total_reward

    def sample_batch(self):

        with self.lock:

            allocs = []
            ids = []
            actions = []
            rewards = []
            states = []
            idxs = []

            for _ in range(self.batch_size):
                buffer_idx = random.randrange(0, self.size)
                time_idx = random.randrange(0, self.buffer[buffer_idx].length-self.n_step-self.block_len)
                idxs.append([buffer_idx, time_idx])

                ids.append([
                    get_context(contexts=self.contexts,
                                tickers=self.buffer[buffer_idx].tickers,
                                date=self.buffer[buffer_idx].timestamps[time_idx+t])
                    for t in range(self.block_len+self.n_step)
                ])
                rewards.append([
                    self.buffer[buffer_idx].rewards[time_idx+t:time_idx+t+self.n_step]
                    for t in range(self.block_len)
                ])
                allocs.append(self.buffer[buffer_idx].allocs[time_idx:time_idx+self.block_len+self.n_step])
                actions.append(self.buffer[buffer_idx].actions[time_idx:time_idx+self.block_len+self.n_step])
                states.append(self.buffer[buffer_idx].states[time_idx])

            ids, bert_targets = mask_ids(ids)

            allocs = torch.tensor(np.stack(allocs)).view(self.batch_size, self.block_len+self.n_step, 1)
            ids = torch.tensor(np.stack(ids)).view(self.batch_size, self.block_len+self.n_step, 501)
            actions = torch.tensor(np.stack(actions)).view(self.batch_size, self.block_len+self.n_step, 1)
            bert_targets = torch.tensor(np.stack(bert_targets)).view(self.batch_size, self.block_len+self.n_step, 501)
            states = torch.tensor(np.stack(states)).view(self.batch_size, self.state_len, self.d_model)

            rewards = torch.tensor(np.sum(np.array(rewards) * self.gamma, axis=2),
                                   dtype=torch.float32
                                   ).view(self.batch_size, self.block_len, 1)

            allocs = allocs.transpose(0, 1).to(torch.float32)
            ids = ids.transpose(0, 1).to(torch.int32)
            actions = actions.transpose(0, 1).unsqueeze(2).to(torch.float32)
            rewards = rewards.transpose(0, 1).to(torch.float32)
            bert_targets = bert_targets.transpose(0, 1).to(torch.int64)
            states = states.to(torch.float32)

            assert allocs.shape == (self.block_len+self.n_step, self.batch_size, 1)
            assert ids.shape == (self.block_len+self.n_step, self.batch_size, 501)
            assert actions.shape == (self.block_len+self.n_step, self.batch_size, 1, 1)
            assert rewards.shape == (self.block_len, self.batch_size, 1)
            assert bert_targets.shape == (self.block_len+self.n_step, self.batch_size, 501)
            assert states.shape == (self.batch_size, self.state_len, self.d_model)

            block = Block(allocs=allocs,
                          ids=ids,
                          actions=actions,
                          rewards=rewards,
                          bert_targets=bert_targets,
                          states=states,
                          idxs=idxs
                          )

        return block

    def update_priorities(self, idxs, states, loss, bert_loss):
        """
        :param idxs: List[List[buffer_idx, time_idx]]
        :param states: Array[batch_size, block_len+n_step, state_len, d_model]
        :param loss: float
        :param bert_loss: float
        """
        assert states.shape == (self.batch_size, self.block_len+self.n_step, self.state_len, self.d_model)

        with self.lock:

            # update new state for each sample in batch
            for idx, state in zip(idxs, states):
                buffer_idx, time_idx = idx

                self.buffer[buffer_idx].states[time_idx:time_idx+self.block_len+self.n_step] = state

            # log
            self.logger.total_updates += 1
            self.logger.loss = loss
            self.logger.bert_loss = bert_loss

    def log(self):

        with self.lock:
            self.logger.print()


class LocalBuffer:

    def __init__(self):
        self.alloc_buffer = []
        self.timestamp_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.state_buffer = []

    def add(self, alloc, timestamp, action, reward, state):
        """
        :param alloc:     float
        :param timestamp: datetime.datetime
        :param action:    float
        :param reward:    float
        :param state:     Array[1, state_len, d_model]
        :return:
        """
        state = state.squeeze(0)

        self.alloc_buffer.append(alloc)
        self.timestamp_buffer.append(timestamp)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def finish(self, tickers, total_reward, total_time):
        """
        :param tickers: List[2]
        :param total_reward: float
        :param total_time: float

        Note: total_reward could be different from reward since it might not be normalized
        """
        tickers = np.stack(tickers)

        allocs = np.stack(self.alloc_buffer)
        timestamps = np.stack(self.timestamp_buffer)
        actions = np.stack(self.action_buffer)
        rewards = np.stack(self.reward_buffer)
        states = np.stack(self.state_buffer)
        length = len(allocs)

        self.alloc_buffer.clear()
        self.timestamp_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.state_buffer.clear()

        return Episode(tickers=tickers,
                       allocs=allocs,
                       timestamps=timestamps,
                       actions=actions,
                       rewards=rewards,
                       states=states,
                       length=length,
                       total_reward=total_reward,
                       total_time=total_time
                       )

