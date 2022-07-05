import numpy as np
from UCB_Learner import UCB_Learner


prices = np.array([[7, 5, 3, 1],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7]])

learner= UCB_Learner(prices)

# idx = np.zeros(5, dtype=int)
# print(idx)
# means = np.zeros(prices.shape)
# widths = np.ones(prices.shape) * 2 

# print(np.argmax((means[1, :]+widths[1 :])*prices[1, :]))
# for i in range(5):
#     idx[i] = np.argmax((means[i, :]+widths[i, :])*prices[i, :])
# print(idx)


pulled_arms= learner.pull_prices()
rewards= np.array([[2,4],
                  [3,4],
                  [4,4],
                  [5,7],
                  [6,8]])
# print(pulled_arms.shape)
# pulled_arms = np.ones(5)
# print(pulled_arms)

#Non avendo voglia di fare una funzione per i reward, copy-paste invece del for
learner.update(pulled_arms, rewards)

pulled_arms= learner.pull_prices()
rewards= np.array([[4,6],
                  [5,5],
                  [4,4],
                  [2,7],
                  [5,9]])
learner.update(pulled_arms, rewards)


pulled_arms= learner.pull_prices()
rewards= np.array([[4,6],
                  [5,5],
                  [4,4],
                  [2,7],
                  [5,9]])
learner.update(pulled_arms, rewards)



pulled_arms= learner.pull_prices()
rewards= np.array([[4,6],
                  [5,5],
                  [4,4],
                  [2,7],
                  [5,9]])
learner.update(pulled_arms, rewards)
print((learner.means+learner.widths)*prices)


pulled_arms= learner.pull_prices()
rewards= np.array([[1,8],
                  [8,8],
                  [4,4],
                  [7,7],
                  [5,9]])
learner.update(pulled_arms, rewards)
print(learner.means)