import numpy as np
from GreedyLearner import Greedy_Learner

prices = np.array([[1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7]])

conversion_rates = np.array([[0.2, 0.3, 0.1, 0.4],
                            [0.1, 0.4, 0.2, 0.3],
                            [0.6, 0.1, 0.1, 0.2],
                            [0.1, 0.1, 0.1, 0.7],
                            [0.2, 0.2, 0.5, 0.1]])

learner = Greedy_Learner(prices, conversion_rates)
print(learner.pull_prices())