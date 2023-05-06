

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import read_prices

import warnings
warnings.filterwarnings("ignore")


class Env:
    def __init__(self,
                 tickers,
                 render,
                 start,
                 end,
                 repeat):
        """
        :param tickers:            List[string]
        :param render:             bool
        :param start:              string
        :param end:                string
        :param repeat:             int

        self.alloc:                initialized in self.reset() indicating where the capital is going
                                   0.5  = 50/50 allocation
                                   0    = 100/0 allocation
                                   1    = 0/100 allocation

        state = (first stock allocation percent, date of current timestep)
        """
        self.tickers = tickers

        self.prices = read_prices(tickers=self.tickers,
                                  start=pd.Timestamp(start),
                                  end=pd.Timestamp(end),
                                  repeat=repeat
                                  )

        # ----------------------------

        self.prices["AAPL"] = 100 + np.arange(len(self.prices.index))
        self.prices["AMZN"] = np.full(len(self.prices.index), fill_value=100)

        print(self.prices)

        self.temp_prices = self.prices.sample(n=2, axis='columns')
        self.temp_prices = self.temp_prices.dropna()
        while len(self.temp_prices) < 10:
            self.temp_prices = self.prices.sample(n=2, axis='columns')
            self.temp_prices = self.temp_prices.dropna()

        self.temp_index = self.temp_prices.index.tolist()
        self.temp_timesteps = len(self.temp_index)
        self.temp_prices = self.temp_prices.reset_index(drop=True)
        self.temp_tickers = self.temp_prices.columns.get_level_values(0).tolist()

        # -----------------------------

        self.render = render
        if render:
            self.render_class = Render()

    def reset(self):
        self.temp_prices = self.prices.sample(n=2, axis='columns')
        self.temp_prices = self.temp_prices.dropna()
        while len(self.temp_prices) < 10:
            self.temp_prices = self.prices.sample(n=2, axis='columns')
            self.temp_prices = self.temp_prices.dropna()

        self.temp_index = self.temp_prices.index.tolist()
        self.temp_timesteps = len(self.temp_index)
        self.temp_prices = self.temp_prices.reset_index(drop=True)
        self.temp_tickers = self.temp_prices.columns.get_level_values(0).tolist()

        self.time = 0
        self.alloc = 0.5

        if self.render:
            self.render_class.reset()

        return (self.alloc, self.temp_index[self.time]), 0, False, self.temp_tickers

    def step(self, action, action_scale=10):
        if self.time >= self.temp_timesteps-1:
            return (self.alloc, self.temp_index[self.time]), 0, True, self.temp_tickers

        self.alloc += (action / action_scale)
        self.alloc = min(max(self.alloc, 0), 1)

        prev_prices = self.temp_prices.loc[self.time].to_numpy()
        self.time += 1
        curr_prices = self.temp_prices.loc[self.time].to_numpy()

        # change of prices represented as % - 0 means unchanged
        change = (curr_prices / prev_prices) - 1
        # (alloc1 * change1) + (alloc2 * change2) = total change
        reward = (self.alloc * change[0]) + ((1-self.alloc) * change[1])

        if self.render:
            self.render_class.step(reward, change[0], change[1])

        return (self.alloc, self.temp_index[self.time]), reward, False, self.temp_tickers

    def get_benchmark(self):
        avg_reward = 0.
        stock1_reward = 0.
        stock2_reward = 0.

        for t in range(1, self.temp_timesteps-1):
            prev_prices = self.temp_prices.loc[t-1].to_numpy()
            curr_prices = self.temp_prices.loc[t].to_numpy()
            change = (curr_prices / prev_prices) - 1

            avg_reward += 0.5 * change[0] + 0.5 * change[1]
            stock1_reward += change[0]
            stock2_reward += change[1]

        return avg_reward, stock1_reward, stock2_reward

    def normalize_reward(self, total_reward):
        _, reward1, reward2 = self.get_benchmark()
        min_reward = min(reward1, reward2)
        max_reward = max(reward1, reward2)
        normalized_reward = (total_reward - min_reward) / (max_reward - min_reward)

        return normalized_reward

    def render_episode(self):
        if self.render:
            self.render_class.render()


class Render:
    eq_r = 1
    eq_c1 = 1
    eq_c2 = 1

    rewards = [1]
    change1 = [1]
    change2 = [1]

    def __init__(self):
        self.reset()

    def reset(self):
        self.eq_r = 1
        self.eq_c1 = 1
        self.eq_c2 = 1

        self.rewards = [1]
        self.change1 = [1]
        self.change2 = [1]

    def step(self, r, c1, c2):
        self.eq_r *= (r + 1)
        self.eq_c1 *= (c1 + 1)
        self.eq_c2 *= (c2 + 1)

        self.rewards.append(self.eq_r)
        self.change1.append(self.eq_c1)
        self.change2.append(self.eq_c2)

    def render(self):
        plt.clf()

        plt.plot(self.change1, c="blue")
        plt.plot(self.change2, c="blue")
        plt.plot(self.rewards, c="red")

        plt.pause(0.00001)
