from matplotlib import use
import numpy as np
import itertools
from Simulator import Simulator
from MonteCarlo_sampling import  MC_sampling
from parameters_generation_utils import alpha_generation, prob_matrix_generation, MC_num_iterations
from UCB_Learner import UCB_Learner
from TS_Learner import TS_Learner
from Number_of_sold_items_estimator import Number_of_sold_items_estimator

#PARAMETER INITIALIZATION

seed = 15
np.random.seed(seed)

prices = np.array([[1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [1, 2, 3, 4]])

conversion_rates = np.array([[[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]],

                             [[0.5, 0.4, 0.3, 0.2],
                              [0.5, 0.4, 0.3, 0.2],
                              [0.5, 0.4, 0.3, 0.2],
                              [0.5, 0.4, 0.3, 0.2],
                              [0.5, 0.4, 0.3, 0.2]],

                             [[0.1, 0.09, 0.08, 0.07],
                              [0.1, 0.09, 0.08, 0.07],
                              [0.1, 0.09, 0.08, 0.07],
                              [0.1, 0.09, 0.08, 0.07],
                              [0.1, 0.09, 0.08, 0.07]]])

n_items_to_buy_distr = np.array([[[19, 2],
                                  [19, 2],
                                  [19, 2],
                                  [19, 2],
                                  [19, 2]],

                                 [[10, 2],
                                  [10, 2],
                                  [10, 2],
                                  [10, 2],
                                  [10, 2]],

                                 [[1, 2],
                                  [2, 2],
                                  [1, 2],
                                  [2, 2],
                                  [1, 2]]])

print(n_items_to_buy_distr[:][0][0])
primary_to_secondary_mapping = np.array([[1,2],
                                         [2,3],
                                         [3,4],
                                         [4,0],
                                         [0,1]])

feature_1_dist = 0.2
feature_2_dist = 0.2

lambda_param = 0.5

#the first alpha is alpha_0
#parameters for the dirichlet that samples the alphas
alpha_parameters = [[2,2,3,4,5,6],
                    [2,2,3,4,5,6],
                    [2,2,3,4,5,6]]

prob_matrix = prob_matrix_generation(primary_to_secondary_mapping, lambda_param)
print('Probability matrix: \n', prob_matrix)

np.random.seed(None)

bandit = UCB_Learner(prices)
items_sold_estimator = Number_of_sold_items_estimator(5, 3)

days = 30
users = 50
n_simulations = 10

opt_per_starting_point = np.zeros((3,5))

#Estimate activation probabilities with MonteCarlo sampling for each user class
activation_probs = []
for user_class in range(prob_matrix.shape[0]):
    estimator = MC_sampling(prob_matrix[user_class])
    activation_probs.append(estimator.estimate_activ_prob( MC_num_iterations(prob_matrix[user_class])))
    print('Activation probabilities: ',activation_probs)

#Create all possible combinationsto of price per item to be evaluated
possible_arms_indeces = np.arange(prices.shape[1])
combinations = []
for comb in itertools.product(possible_arms_indeces, repeat=len(possible_arms_indeces)+1):
  combinations.append(comb)

#compute opts per starting point
for user_class in range(0, conversion_rates.shape[0]):
    for starting_point in range(0, prices.shape[0]):
        combinations_rewards = []
        for comb in combinations:
            cur_sum = 0
            for item, arm in enumerate(comb):
                cur_sum += prices[item][arm] * conversion_rates[user_class][item][arm] * n_items_to_buy_distr[user_class][item][0] * (activation_probs[user_class][starting_point][item])
            combinations_rewards.append(cur_sum)
        opt_per_starting_point[user_class][starting_point] = np.max(combinations_rewards)
print(opt_per_starting_point)

#test simulation
S = Simulator(days,
              users,
              n_simulations,
              alpha_parameters,
              seed,
              bandit,
              False, #True if the number of sold items is uncertain
              items_sold_estimator,
              False, #True if context has to be used
              prices,
              prob_matrix,
              feature_1_dist,
              feature_2_dist,
              conversion_rates,
              primary_to_secondary_mapping,
              n_items_to_buy_distr,
              opt_per_starting_point,
              activation_probs)

S.run_simulation(debug=False)
S.plot_cumulative_regret()

'''
#test MonteCarlo algorithm
estimator = MC_sampling(prob_matrix[2])
activation_probs = estimator.estimate_activ_prob(9000)
print('Activation probabilities: ',activation_probs)
'''


