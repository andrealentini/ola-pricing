import numpy as np
from Learner import *

class SWUCB_Learner(Learner):
    def __init__(self, prices, window_size):
        super().__init__(prices)
        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.previous_arms = [0] * prices.shape[0]
        self.window_size = window_size  # length of window in days
        # nr of rewards collected at day t for item and arm
        self.nr_rewards_per_day = [[[] for _ in range(self.n_prices)] for __ in range(self.n_items)]
    
    def pull_prices_activations(self, n_items_to_buy_distr, activation_probs):
        combinations = []
        for item in range(self.prices.shape[0]):
            for arm in range(self.prices.shape[1]):
                combination = self.previous_arms.copy()
                combination[item] = arm
                combinations.append(combination)
        combinations_rewards = []
        for starting_point in range(0, self.prices.shape[0]):
            for comb in combinations:
                cur_sum = 0
                for item, arm in enumerate(comb):
                    cur_sum += self.prices[item][arm] * (self.means[item, arm]+self.widths[item, arm]) * np.mean(n_items_to_buy_distr[:, item, 0], axis=0) * np.mean(np.array(activation_probs)[:,starting_point, item], axis=0)
                combinations_rewards.append(cur_sum)
        combinations = combinations * self.prices.shape[0]
        pulled_arms_idx = combinations[np.argmax(combinations_rewards)]

        for idx in range(self.n_items):
            for idy in range(self.n_prices):
                # effective rewards (not -1) inside the window
                nr_rewards_in_window = sum(self.nr_rewards_per_day[idx][idy][-self.window_size:])
                # rewards for item and arm of days in window, excluding placeholders (-1)
                rewards_per_item_arm_in_window = [reward for reward in self.rewards_per_item_arm[idx][idy][-nr_rewards_in_window:] if reward != -1]
                n = len(rewards_per_item_arm_in_window)
                if n == 0:
                    # pull the arm if it has not been pulled inside the window
                    pulled_arms_idx[idx] = idy
                    break

        self.previous_arms = pulled_arms_idx
        print('Pulled arms: ', pulled_arms_idx)
        return pulled_arms_idx

    
    def reset(self):
        self.__init__(self.prices, self.window_size)

    def update(self, pulled_arms, rewards): #pulled arms è un vettore di dim 5, rewards ha dim 5x(volte che pesco i rewards in una giornata)      
        self.t += 1

        for i in range(self.n_items):
            for j in range(self.n_prices):
                # placeholders (-1) used to express that an arm has not been pulled in the day
                # trick used since nr of rewards in a day is not constant
                # otherwise would have overflow when taking rewards in window
                self.rewards_per_item_arm[i][j] = self.rewards_per_item_arm[i][j] + (rewards[i] if j == pulled_arms[i] else [-1] * len(rewards[i]))
                self.nr_rewards_per_day[i][j].append(len(rewards[i]))
        
        for i in range(self.n_items):
            self.collected_rewards_per_item[i] = self.collected_rewards_per_item[i] + rewards[i]
        
        for i in range(self.n_items):
            if len(rewards[i]) != 0:
                # nr of rewards collected in the days in window
                nr_rewards_in_window = sum(self.nr_rewards_per_day[i][pulled_arms[i]][-self.window_size:])
                # rewards for item and arm of days in window, excluding placeholders (-1)
                rewards_per_item_arm_in_window = [reward for reward in self.rewards_per_item_arm[i][pulled_arms[i]][-nr_rewards_in_window:] if reward != -1]
                self.means[i][pulled_arms[i]] = np.mean(rewards_per_item_arm_in_window)
        
        for idx in range(self.n_items):
            for idy in range(self.n_prices):
                # widths update with window, same as means
                nr_rewards_in_window = sum(self.nr_rewards_per_day[idx][idy][-self.window_size:])
                rewards_per_item_arm_in_window = [reward for reward in self.rewards_per_item_arm[idx][idy][-nr_rewards_in_window:] if reward != -1]
                n = len(rewards_per_item_arm_in_window)
                if n > 0:
                    self.widths[idx][idy] = np.sqrt(2*np.log(self.t)/n)
                else:
                    self.widths[idx][idy] = np.inf