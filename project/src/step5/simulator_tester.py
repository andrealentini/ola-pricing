from matplotlib import use
import numpy as np
import itertools
from parameters_generation_utils import alpha_generation, prob_matrix_generation, MC_num_iterations

from Threaded_Estimate_Probabilities import Probabilities_Estimator

#PARAMETER INITIALIZATION

seed = 15
np.random.seed(seed)

prices = np.array([[1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [1, 2, 3, 4]])

conversion_rates = np.array([[[0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4]],

                             [[0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4]],

                             [[0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4],
                              [0.7, 0.6, 0.5, 0.4]]])

n_items_to_buy_distr = np.array([[[1, 2],
                                  [2, 2],
                                  [1, 2],
                                  [2, 2],
                                  [1, 2]],

                                 [[1, 2],
                                  [2, 2],
                                  [1, 2],
                                  [2, 2],
                                  [1, 2]],

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

feature_1_dist = 0.5
feature_2_dist = 0.5

lambda_param = 0.5

#the first alpha is alpha_0
#parameters for the dirichlet that samples the alphas
alpha_parameters = [[2,2,3,4,5,6],
                    [2,2,3,4,5,6],
                    [2,2,3,4,5,6]]

######### CHANGES START HERE #########

# prob matrix as np array needed for estimation
prob_matrix = np.array(prob_matrix_generation(primary_to_secondary_mapping, lambda_param))

# alphas needed to pass to prob estimation for initial nodes
# 5 nodes needed instead of 6 (competitor case not needed for prob matrix estimation)
# alpha_generation method modified to take number of nodes "n". Default is n=6 so it works with other parts
# see the method in parameters_generation_utils in this folder
alphas = alpha_generation(np.array(alpha_parameters)[:, 1:], seed=seed, n=5)

# object that estimates the prob matrix
prob_estimator = Probabilities_Estimator(prob_matrix, alphas, np.arange(0,5,1))

# estimated probability matrix, to override original prob matrix if needed
estimated_prob_matrix = prob_estimator.estimate_probabilities()

print('Original Probability Matrix: \n', np.mean(prob_matrix, axis=0))
print('Estimated Probability Matrix: \n', estimated_prob_matrix)

exit(0)

######### CHANGES END HERE #########

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


