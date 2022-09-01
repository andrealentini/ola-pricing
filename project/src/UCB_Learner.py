import numpy as np
from Learner import *
import itertools

# Implementation of the specific Multi Armed Bandit algorithm UCB (Upper Confidence Bound), derived from general Learner class 
class UCB_Learner(Learner):

    # UCB needs to keep track of means and confidence bounds of the expected reward for each arm (both initialized with zeros)
    # previous_arms : arms selected at the previous call of the bandit are used to make an estimation of the new best arms to pick and speed up the pull of new candidate prices
    def __init__(self, prices, bandit_split=None):
        super().__init__(prices, bandit_split)
        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.previous_arms = [0] * prices.shape[0]
        
    '''
    def pull_prices(self): #to pick the 5 arms, 1 for each item, return the indexes inside prices matrix
        idx = np.zeros(self.n_items, dtype=int)
       #todo considerare anche il numero di item venduti per ogni item
        for i in range(self.n_items):
            idx[i] = np.argmax((self.means[i, :]+self.widths[i, :])*self.prices[i, :])
        return idx
    '''
    
    # Method that pulls new candidate arms before the collection of the rewards. Arms selected optimize the expected rewards estimated with confidence bounds,
    # given the distribution of the number of items bought for each item and the activation probabilities
    # estimated : when True it uses previous arms instead of all possible combinations to optimize the expected rewards of new pulled arms, with a notable speedup in computation
    def pull_prices_activations(self, n_items_to_buy_distr, activation_probs, estimated=True):

        combinations = []

        if estimated==False:
            idx = np.zeros(self.n_items, dtype=int)
            possible_arms_indeces = np.arange(self.prices.shape[1])
            for comb in itertools.product(possible_arms_indeces, repeat=len(possible_arms_indeces)+1):
                combinations.append(comb)

        else:
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

        return pulled_arms_idx

    # Reinitialize the bandit
    def reset(self):
        self.__init__(self.prices, self.bandit_split)

    # This method extends the Learner update to also update means and bounds of each arm, of each item
    def update(self, pulled_arms, rewards):
        
        super().update(pulled_arms, rewards)

        for i in range(self.n_items):
            if len(rewards[i]) != 0:
                self.means[i][pulled_arms[i]] = np.mean(self.rewards_per_item_arm[i][pulled_arms[i]])
        
        for idx in range(self.n_items):
            for idy in range(self.n_prices):
                n = len(self.rewards_per_item_arm[idx][idy])
                if n > 0:
                    self.widths[idx][idy] = np.sqrt(2*np.max(self.prices[idx, :])*np.log(self.t)/n)
                else:
                    self.widths[idx][idy] = np.inf
