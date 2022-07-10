import numpy as np
from TS_Learner import TS_Learner

prices = np.array([[7, 5, 3, 1],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7]])

learner = TS_Learner(prices)

pulled_arms= learner.pull_prices()
print("Pulled arms: ", pulled_arms)

rewards= np.array([[1, 0],
                  [0,1],
                  [0,1],
                  [1,0],
                  [0,0]])

learner.update(pulled_arms, rewards)
# print(learner.beta_parameteres)

pulled_arms= learner.pull_prices()
print("Pulled arms: ", pulled_arms)

rewards= np.array([[0, 1],
                  [0,1],
                  [0,1],
                  [0,1],
                  [0,1]])

learner.update(pulled_arms, rewards)

print(learner.beta_parameteres)