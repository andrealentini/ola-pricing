import numpy as np
from Learner import *

class UCB_Learner(Learner):
    def __init__(self, prices):
        super().__init__(prices)
        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        
    
    def pull_prices(self): #to pick the 5 arms, 1 for each item, return the indexes inside prices matrix
        idx = np.zeros(self.n_items, dtype=int)
       #todo considerare anche il numero di item venduti per ogni item
        for i in range(self.n_items):
            idx[i] = np.argmax((self.means[i, :]+self.widths[i, :])*self.prices[i, :])
        return idx
    
    def reset(self):
        self.__init__(self.prices)

    def update(self, pulled_arms, rewards): #pulled arms è un vettore di dim 5, rewards ha dim 5x(volte che pesco i rewards in una giornata)
        
        super().update(pulled_arms, rewards)

        # i mi dice qual è l'item, pulled_arm[i] mi dice l'arm pullato per l'item i
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
