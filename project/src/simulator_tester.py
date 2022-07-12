import numpy as np
from Simulator import Simulator
from MonteCarlo_sampling import  MC_sampling
from parameters_generation_utils import alpha_generation, prob_matrix_generation
from UCB_Learner import UCB_Learner
from GreedyLearner import Greedy_Learner

#PARAMETER INITIALIZATION

seed = 15
np.random.seed(seed)

prices = np.array([[1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7]])

conversion_rates = np.array([[[0.7, 0.5, 0.9, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1]],

                             [[0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1]],

                             [[0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1],
                              [0.7, 0.5, 0.3, 0.1]]])

n_items_to_buy_distr = np.array([[[5, 2],
                                  [4, 2],
                                  [3, 2],
                                  [3, 2],
                                  [4, 2]],

                                 [[5, 2],
                                  [4, 2],
                                  [3, 2],
                                  [3, 2],
                                  [4, 2]],

                                 [[5, 2],
                                  [4, 2],
                                  [3, 2],
                                  [3, 2],
                                  [4, 2]]])


primary_to_secondary_mapping = np.array([[1,2],
                                         [2,3],
                                         [3,4],
                                         [4,0],
                                         [0,1]])

feature_1_dist = 0.5
feature_2_dist = 0.5

lambda_param = 0.5

#the first alpha is alpha_0
#parameters for the dirichlet that samples the alphas
alpha_parameters = [[1,2,3,40,5,6],
                    [10,2,3,4,5,6],
                    [1,2,3,40,5,6]]

prob_matrix = prob_matrix_generation(primary_to_secondary_mapping, lambda_param)
print('Probability matrix: \n', prob_matrix)

np.random.seed(None)

bandit = Greedy_Learner(prices, conversion_rates[0], n_items_to_buy_distr[0])

days = 10
users = 1000
n_simulations = 10

opt = np.zeros((3,5))

for user_class in range(0, conversion_rates.shape[0]):
    for item in range(0, prices.shape[0]):
        opt[user_class][item] = np.max([price * conv for price, conv in zip(prices[item], conversion_rates[user_class][item])])
        opt[user_class][item] = opt[user_class][item] * n_items_to_buy_distr[user_class][item][0]

#test simulation
S = Simulator(days,
              users,
              n_simulations,
              alpha_parameters,
              seed,
              bandit,
              prices,
              prob_matrix,
              feature_1_dist,
              feature_2_dist,
              conversion_rates,
              primary_to_secondary_mapping,
              n_items_to_buy_distr,
              opt)

S.run_simulation(debug=False)
S.plot_cumulative_regret()

'''
#test MonteCarlo algorithm
estimator = MC_sampling(prob_matrix[2])
activation_probs = estimator.estimate_activ_prob(9000)
print('Activation probabilities: ',activation_probs)
'''