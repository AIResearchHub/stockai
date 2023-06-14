import threading

import pandas as pd
import matplotlib.pyplot as plt

from utils import read_prices

import warnings
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from utils import read_prices

warnings.filterwarnings("ignore")

class Env:
    """
    state = (allocation value (0, 1), date of current time step (datetime.datetime))


    alloc:      initialized in reset() indicating where the capital is going
                0.5  = 50/50 allocation
                0    = 100/0 allocation
                1    = 0/100 allocation


    Parameters:
    tickers (List[String]): list of tickers
    render (bool): If true render real time plot during training with matplotlib
    start (String): start date in format "2020-01-01"
    end (String): end date in format "2020-01-01"
    repeat (int): If repeat, then prices are duplicated "repeat" times to look at more context at each date and time

    """

    def __init__(self,
                 tickers,
                 render,
                 start,
                 end,
                 repeat):
        self.tickers = tickers

        self.prices = read_prices(tickers=self.tickers,
                                  start=pd.Timestamp(start),
                                  end=pd.Timestamp(end),
                                  repeat=repeat
                                  )

        # ----------------------------
        #
        # self.prices["AAPL"] = 100 + np.arange(len(self.prices.index))
        # self.prices["AMZN"] = np.full(len(self.prices.index), fill_value=100)
        #
        # print(self.prices)
        #
        # self.temp_prices = self.prices.sample(n=2, axis='columns')
        # self.temp_prices = self.temp_prices.dropna()
        # while len(self.temp_prices) < 10:
        #     self.temp_prices = self.prices.sample(n=2, axis='columns')
        #     self.temp_prices = self.temp_prices.dropna()
        #
        # self.temp_index = self.temp_prices.index.tolist()
        # self.temp_timesteps = len(self.temp_index)
        # self.temp_prices = self.temp_prices.reset_index(drop=True)
        # self.temp_tickers = self.temp_prices.columns.get_level_values(0).tolist()
        #
        # -----------------------------

        self.render = render
        if render:
            self.render_class = Render()

    def reset(self):
        """
        This function is called before every episode, it re-samples to tickers,
        prepare the prices, reset time and alloc constant, and reset render class

        Returns:
        state (float, datetime.datetime): initial state
        reward (float): always 0 since episode just started
        done (bool): always False since episode just started
        temp_tickers (List[2]): 2 newly sampled tickers

        """
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
        """
        This function is called before at every time step, if last time step then return done = True,
        allocation is updated according to action, and reward is computed according to prices,
        and render class step is called.

        Parameters:
        action (float): Action value (0, 1)
        action_scale (float = 10): Value to divide action by, the smaller the value, the longer it takes to change allocation

        Returns:
        state (float, datetime.datetime): initial state
        reward (float): always 0 since episode just started
        done (bool): always False since episode just started
        temp_tickers (List[2]): 2 newly sampled tickers

        """
        if self.time >= self.temp_timesteps - 1:
            return (self.alloc, self.temp_index[self.time]), 0, True, self.temp_tickers

        self.alloc += (action / action_scale)
        self.alloc = min(max(self.alloc, 0), 1)

        prev_prices = self.temp_prices.loc[self.time].to_numpy()
        self.time += 1
        curr_prices = self.temp_prices.loc[self.time].to_numpy()

        # change of prices represented as % - 0 means unchanged
        change = (curr_prices / prev_prices) - 1
        # (alloc1 * change1) + (alloc2 * change2) = total change
        reward = (self.alloc * change[0]) + ((1 - self.alloc) * change[1])

        if self.render:
            self.render_class.step(reward, change[0], change[1])

        return (self.alloc, self.temp_index[self.time]), reward, False, self.temp_tickers

    def get_benchmark(self):
        """
        In order to determine if a reward is above average, average reward, stock 1 reward, and stock 2 rewards
        are computed.

        Average reward is 0.5 allocation
        Stock 1 reward is 0.0 allocation
        Stock 2 reward is 1.0 allocation

        Returns:
        avg_reward (float): average reward
        stock1_reward (float): holding stock 1 reward
        stock2_reward (float): holding stock 2 reward

        """
        avg_reward = 0.
        stock1_reward = 0.
        stock2_reward = 0.

        for t in range(1, self.temp_timesteps - 1):
            prev_prices = self.temp_prices.loc[t - 1].to_numpy()
            curr_prices = self.temp_prices.loc[t].to_numpy()
            change = (curr_prices / prev_prices) - 1

            avg_reward += 0.5 * change[0] + 0.5 * change[1]
            stock1_reward += change[0]
            stock2_reward += change[1]

        return avg_reward, stock1_reward, stock2_reward

    def normalize_reward(self, total_reward):
        """
        This function calls get_benchmark() to get stock1 and stock2 reward
        normalized reward is then computed from turning it into an interval
        where min(stock1, stock2) is 0 and max(stock1, stock2) is 1

        Returns:
        normalized_reward (float): normalized reward of total reward
        """
        _, reward1, reward2 = self.get_benchmark()
        min_reward = min(reward1, reward2)
        max_reward = max(reward1, reward2)
        normalized_reward = (total_reward - min_reward) / (max_reward - min_reward)

        return normalized_reward

    def render_episode(self):
        """Render the episode when episode is finished"""
        if self.render:
            self.render_class.render()


class Render:
    """
    This simple helper class is used inside Env to keep track of the allocations
    to be used for rendering at the end of each episode in matplotlib during training

    """
    eq_r = 1
    eq_c1 = 1
    eq_c2 = 1

    rewards = [1]
    change1 = [1]
    change2 = [1]

    def __init__(self):
        self.reset()

    def reset(self):
        """Called at the start of each episode to reset the variables"""
        self.eq_r = 1
        self.eq_c1 = 1
        self.eq_c2 = 1

        self.rewards = [1]
        self.change1 = [1]
        self.change2 = [1]

    def step(self, r, c1, c2):
        """Called at each time step to record the allocations"""
        self.eq_r *= (r + 1)
        self.eq_c1 *= (c1 + 1)
        self.eq_c2 *= (c2 + 1)

        self.rewards.append(self.eq_r)
        self.change1.append(self.eq_c1)
        self.change2.append(self.eq_c2)

    def is_main_thread(self):
        return threading.current_thread() == threading.main_thread()

    def render(self):
        """Called at the end of the episode to render the plot"""
        plt.clf()

        plt.plot(self.change1, c="blue")
        plt.plot(self.change2, c="blue")
        plt.plot(self.rewards, c="red")

        # Only call render if we're in the main thread
        if self.is_main_thread():
            plt.pause(0.00001)
