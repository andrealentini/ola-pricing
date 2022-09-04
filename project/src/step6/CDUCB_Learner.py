import numpy as np
from Learner import *
from cusum import CUSUM

class CDUCB_Learner(Learner):
    def __init__(self, prices, M=25, eps=0.05, h=10, alpha=0.01):
        super().__init__(prices)
        self.M = M
        self.eps = eps
        self.h = h
        self.alpha = alpha

        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.previous_arms = [0] * prices.shape[0]
        
        # change detectors for each item and arm
        self.change_detection = [[CUSUM(M, eps, h) for _ in range(self.n_prices)] for item in range(self.n_items)]
        # rewards collected before a change for each item and arm
        self.valid_rewards_per_item_arm = [[[] for _ in range(self.n_prices)] for item in range(self.n_items)]
        # list of detections
        self.detections = [[[] for _ in range(self.n_prices)] for item in range(self.n_items)]
    
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
        self.previous_arms = pulled_arms_idx
        # print("Pulled arms: ", pulled_arms_idx)
        return pulled_arms_idx
    
    def reset(self):
        self.__init__(self.prices, self.M, self.eps, self.h, self.alpha)
    
    def update(self, pulled_arms, rewards):
        self.t += 1

        for i in range(self.n_items):
            if self.change_detection[i][pulled_arms[i]].update(rewards[i]):
                # If a change has been detected for item and arm, empty the rewards and reset the detector
                print(f"Change detected for item {i} and arm {pulled_arms[i]} at time {self.t}")
                self.detections[i][pulled_arms[i]].append(self.t)
                self.valid_rewards_per_item_arm[i][pulled_arms[i]] = []
                self.change_detection[i][pulled_arms[i]].reset()

            self.rewards_per_item_arm[i][pulled_arms[i]] = self.rewards_per_item_arm[i][pulled_arms[i]] + rewards[i]
            self.valid_rewards_per_item_arm[i][pulled_arms[i]] = self.valid_rewards_per_item_arm[i][pulled_arms[i]] + rewards[i]
            self.collected_rewards_per_item[i] = self.collected_rewards_per_item[i] + rewards[i]
            if len(rewards[i]) != 0:
                self.means[i][pulled_arms[i]] = np.mean(self.valid_rewards_per_item_arm[i][pulled_arms[i]])
        
        for idx in range(self.n_items):
            total_valid_samples = sum([len(x) for x in self.valid_rewards_per_item_arm[idx]])
            for idy in range(self.n_prices):
                n = len(self.valid_rewards_per_item_arm[idx][idy])
                if n > 0:
                    self.widths[idx][idy] = np.sqrt((2 * np.log(total_valid_samples) / n))
                else:
                    self.widths[idx][idy] = np.inf