import numpy as np
from Simulator import Simulator
from MonteCarlo_sampling import  MC_sampling
from parameters_generation_utils import alpha_generation, prob_matrix_generation

#PARAMETER INITIALIZATION

seed = 15
np.random.seed(seed)

prices = np.array([[1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7],
                   [1, 3, 5, 7]])

conversion_rates = np.array([[[0.7, 0.5, 0.3, 0.1],
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
alphas = alpha_generation(alpha_parameters, seed=seed)
print('Alphas: \n', alphas)

print('\n')

prob_matrix = prob_matrix_generation(primary_to_secondary_mapping, lambda_param)
print('Probability matrix: \n', prob_matrix)

np.random.seed(None)

days = 1
users = 1

#test simulation
S = Simulator(days,
              users,
              prices,
              prob_matrix,
              alphas,
              feature_1_dist,
              feature_2_dist,
              conversion_rates,
              primary_to_secondary_mapping,
              n_items_to_buy_distr)

S.run_simulation(debug=True)

#test MonteCarlo algorithm
estimator = MC_sampling(prob_matrix[2])
activation_probs = estimator.estimate_activ_prob(1000)
print('Activation probabilities: ',activation_probs)
