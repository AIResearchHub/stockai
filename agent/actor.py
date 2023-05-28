

import numpy as np
import time

from .replay_buffer import LocalBuffer
from env import Env


class Actor:
    """
    Class to be asynchronously run by Learner, use self.run() for main training
    loop. This class creates a local buffer to store data before sending completed
    Episode to Learner through rpc

    Parameters:
    learner_rref (RRef): Learner RRef to reference the learner
    tickers (list): A list of tickers e.g. ["AAPL", "GOOGL", "BAC"]
    d_model (int): The dimensions of the model
    state_len (int): The length of the recurrent state [batch_size, state_len, d_model]
    """

    def __init__(self,
                 learner_rref,
                 tickers,
                 d_model,
                 state_len
                 ):
        """"""
        # np.random.seed(0)

        self.learner_rref = learner_rref

        self.d_model = d_model
        self.state_len = state_len

        self.env = Env(tickers=tickers,
                       render=True,
                       start="2020-11-01",  # 2010
                       end="2021-01-01",
                       repeat=1
                       )

        self.local_buffer = LocalBuffer()

    def get_action(self, alloc, timestamp, tickers, state):
        """
        Uses learner RRef and rpc async to call queue_request to get action
        from learner.

        Parameters:
        alloc (float): allocation parameter from env
        timestamp (datetime.datetime): date of the current timestep
        tickers (List[2]): List of size 2 containing tickers e.g. ["AAPL", "GOOGL"]
        state (np.array): numpy array with shape (batch_size, state_len, d_model)

        Returns:
        Future() object that when used with .wait(), halts until value is ready from
        the learner. It returns action(float) and state(np.array)

        """
        future_action = self.learner_rref.rpc_async().queue_request(alloc,
                                                                    timestamp,
                                                                    tickers,
                                                                    state
                                                                    )
        return future_action

    def return_episode(self, episode):
        """
        Once episode is completed return_episode uses learner_rref and rpc_async
        to call return_episode to return Episode object to learner for training.

        Parameters:
        episode (Episode)

        Returns:
        future_await (Future): halts with .wait() until learner is finished
        """
        future_await = self.learner_rref.rpc_async().return_episode(episode)
        return future_await

    def run(self):
        """
        Main actor training loop, calls queue_request to get action and
        return_episode to return finished episode
        """

        while True:
            (alloc, timestamp), total_reward, done, tickers = self.env.reset()
            state = np.random.randn(1, self.state_len, self.d_model)

            start = time.time()
            while not done:
                action, new_state = self.get_action(alloc, timestamp, tickers, state).wait()

                (new_alloc, new_timestamp), reward, done, tickers = self.env.step(action)

                self.local_buffer.add(alloc, timestamp, action, reward, state)

                alloc = new_alloc
                timestamp = new_timestamp
                state = new_state

                total_reward += reward

            total_reward = self.env.normalize_reward(total_reward)
            episode = self.local_buffer.finish(tickers, total_reward, time.time()-start)
            self.return_episode(episode).wait()

            self.env.render_episode()
