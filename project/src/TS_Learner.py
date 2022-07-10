from Learner import *
import numpy as np


class TS_Learner(Learner):

    def __init__(self, prices):
        super().__init__(prices)
        self.prices = prices

        self.beta_parameteres = np.ones((self.n_items, self.n_prices, 2)) #matrice 5x4x2

    def pull_prices(self):
        idx = np.zeros(self.n_items, dtype=int)

        for i in range(self.n_items):
            idx[i] = np.argmax(np.random.beta(self.beta_parameteres[i, :, 0], self.beta_parameteres[i, :, 1]))
        return idx

    def update(self, pulled_arms, rewards):
        #self.t += 1

        super().update(pulled_arms, rewards)

        for i in range(self.n_items):
            #Non lo so quante colonne ha la matrice rewards in teoria, ma dato che sono più di una, devo iterare anche per j così sommo 1 uno alla volta i singoli rewards
            for j in range(rewards.shape[1]): 
                self.beta_parameteres[i, pulled_arms[i], 0] = self.beta_parameteres[i, pulled_arms[i], 0] + rewards[i, j]
                self.beta_parameteres[i, pulled_arms[i], 1] = self.beta_parameteres[i, pulled_arms[i], 1] + 1.00 - rewards[i, j]
       