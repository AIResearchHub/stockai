

import numpy as np
import time

from .replay_buffer import LocalBuffer
from env import Env


class Actor:

    def __init__(self,
                 learner_rref,
                 n_envs,
                 tickers,
                 d_model,
                 state_len
                 ):
        np.random.seed(0)

        self.learner_rref = learner_rref
        self.n_envs = n_envs

        self.d_model = d_model
        self.state_len = state_len

        self.envs = [Env(tickers=tickers,
                         render=False,
                         start="2021-01-01",
                         end="2023-01-01",
                         repeat=1)
                     for _ in range(n_envs)]

        self.local_buffer = LocalBuffer()

    def get_action(self, alloc, timestamp, tickers, state):
        """
        :param alloc:     float
        :param timestamp: datetime.datetime
        :param tickers:   List[2]
        :param state:     Array[1. state_len, d_model]
        :return:
            Future(
                 action:  float
                 state:   Array[1, state_len, d_model]
            )
        """
        future_action = self.learner_rref.rpc_async().queue_request(alloc,
                                                                    timestamp,
                                                                    tickers,
                                                                    state
                                                                    )
        return future_action

    def return_episode(self, episode):
        """
        sends completed episode back to learner
        """
        self.learner_rref.rpc_async().return_episode(episode)

    def run(self):
        allocs = [None for _ in self.envs]
        timestamps = [None for _ in self.envs]
        states = [None for _ in self.envs]
        total_rewards = [None for _ in self.envs]
        dones = [True for _ in self.envs]
        tickers = [None for _ in self.envs]

        start = time.time()
        while True:
            for i, env in enumerate(self.envs):
                if dones[i]:
                    (allocs[i], timestamps[i]), total_rewards[i], dones[i], tickers[i] = env.reset()
                    states[i] = np.zeros((1, self.state_len, self.d_model))

            actions, new_states = self.get_actions(allocs, timestamps, tickers, states).wait()

            for env, state in zip(self.envs, self.states):
                (new_alloc, new_timestamp), reward, done, tickers = self.env.step(action)

                self.local_buffer.add(alloc, timestamp, action, reward, state)

                alloc = new_alloc
                timestamp = new_timestamp
                state = new_state

                total_reward += reward
