from Learner import *
import numpy as np


class TS_Learner(Learner):

    def __init__(self, prices):
        super().__init__(prices)
        self.prices = prices

        """self.beta_parameteres = np.ones((self.n_items, self.n_prices, 2)) #matrice 5x4x2"""

        #Gaussian-Gamma model (conjugate prior of a normal distribution with unknown mean and precision)
        self.n = np.zeros((self.n_items, self.n_prices))          #collect the number of times an price is pulled, for each item

        self.alphas = np.ones((self.n_items, self.n_prices))      #gamma shape parameters
        self.betas = np.ones((self.n_items, self.n_prices)) * 10  #gamma rate parameters

        self.mu_zero = np.ones((self.n_items, self.n_prices))     #prior (normal estimated mean)
        self.v_zero = self.betas / (self.alphas + 1)              #prior (normal estimated variance)

    def pull_prices(self):
        idx = np.zeros(self.n_items, dtype=int)
        precision = np.zeros((self.n_items, self.n_prices))
        estimated_variance = np.empty((self.n_items, self.n_prices))

        """for i in range(self.n_items):
            idx[i] = np.argmax(np.random.beta(self.beta_parameteres[i, :, 0], self.beta_parameteres[i, :, 1]))"""

        # Gaussian-Gamma model: sample from our estimated gaussian-gamma distribution
        for i in range(self.n_items):
            precision[i] = np.random.gamma(self.alphas[i, :], 1/self.betas[i, :])
            for k in range(precision.shape[1]):
                if precision[i, k] == 0 or self.n[i, k] == 0: precision[i, k] = 0.001
            estimated_variance[i] = 1/precision[i]
            idx[i] = np.argmax(np.random.normal(self.mu_zero[i, :], np.sqrt(estimated_variance[i, :])))

        return idx

    def reset(self):
        self.__init__(self.prices)

    def update(self, pulled_arms, rewards):
        #self.t += 1

        super().update(pulled_arms, rewards)

        """for i in range(self.n_items):
            #Non lo so quante colonne ha la matrice rewards in teoria, ma dato che sono più di una, devo iterare anche per j così sommo 1 uno alla volta i singoli rewards
            for j in range(rewards.shape[1]): 
                self.beta_parameteres[i, pulled_arms[i], 0] = self.beta_parameteres[i, pulled_arms[i], 0] + rewards[i, j]
                self.beta_parameteres[i, pulled_arms[i], 1] = self.beta_parameteres[i, pulled_arms[i], 1] + 1.00 - rewards[i, j]"""

        #When rewards are not binary, we can use different distributions to model out the problem, for instance 
        #under assumption of normal reward with unknown mean and variance, we can use a Gaussian-Gamma conjugate-prior
        for item in range(self.n_items):
            n = len(rewards[item])
            v = self.n[item, pulled_arms[item]]
            for j in range(n): 
                self.alphas[item, pulled_arms[item]] = self.alphas[item, pulled_arms[item]] + n/2
                self.betas[item, pulled_arms[item]] = self.betas[item, pulled_arms[item]] + ((n*v/(v + n)) * (((rewards[item][j] - self.mu_zero[item, pulled_arms[item]])**2)/2))
                # estimate the prior variance: calculate from the gamma hyper-parameters
                self.v_zero[item, pulled_arms[item]] = self.betas[item, pulled_arms[item]] / (self.alphas[item, pulled_arms[item]] + 1)
                # estimate the prior mean: calculate from the collected rewards
                self.mu_zero[item, pulled_arms[item]] = np.array(self.rewards_per_item_arm[item][pulled_arms[item]]).mean()
            self.n[item, pulled_arms[item]] += n
                