from matplotlib import use
import numpy as np
import itertools
from Simulator import Simulator
from MonteCarlo_sampling import  MC_sampling
from parameters_generation_utils import alpha_generation, prob_matrix_generation, MC_num_iterations
from UCB_Learner import UCB_Learner
from TS_Learner import TS_Learner
from Number_of_sold_items_estimator import Number_of_sold_items_estimator

# Code use to launch the project that instantiate and initialize all the needed to make a run

# ================== PROBLEM  MODELING ================== #

seed = 15
np.random.seed(seed)

# Set the value of all candidate prices for each item(rows)
prices = np.array([[1, 2, 3, 4],
                   [1, 2, 3, 4],
                   [2, 3, 4, 5],
                   [2, 3, 4, 5],
                   [3, 4, 5, 6]])

# Set the matrix of conversion rates for each couple item-arm for all the three user classes
conversion_rates = np.array([[[0.85, 0.8, 0.75, 0.7],
                              [0.85, 0.8, 0.75, 0.7],
                              [0.8, 0.75, 0.7, 0.65],
                              [0.8, 0.75, 0.7, 0.65],
                              [0.75, 0.7, 0.65, 0.6]],

                             [[0.5, 0.45, 0.4, 0.35],
                              [0.5, 0.45, 0.4, 0.35],
                              [0.45, 0.4, 0.35, 0.3],
                              [0.45, 0.4, 0.35, 0.3],
                              [0.4, 0.35, 0.3, 0.25]],

                             [[0.2, 0.15, 0.1, 0.05],
                              [0.2, 0.15, 0.1, 0.05],
                              [0.15, 0.1, 0.05, 0.05],
                              [0.15, 0.1, 0.05, 0.05],
                              [0.1, 0.09, 0.08, 0.07]]])

# Set the means and std of the number of items sold for each item for all the user classes
n_items_to_buy_distr = np.array([[[2, 2],
                                  [2, 2],
                                  [2, 2],
                                  [2, 2],
                                  [2, 2]],

                                 [[1.5, 2],
                                  [1.5, 2],
                                  [1.5, 2],
                                  [1.5, 2],
                                  [1.5, 2]],

                                 [[1, 2],
                                  [1, 2],
                                  [1, 2],
                                  [1, 2],
                                  [1, 2]]])

# Set the mapping of the two secondary items showed for each primary item
primary_to_secondary_mapping = np.array([[1,2],
                                         [2,3],
                                         [3,4],
                                         [4,0],
                                         [0,1]])

# Set the distribution of the user features
feature_1_dist = 0.2
feature_2_dist = 0.2

# Set the lambda value (factor that reduce the probability to click on the second secondary item showed in the page of a primary item)
lambda_param = 0.5

# Set parameters for the dirichlet that samples the alphas (probabilities to land on the page of a primary item) for each user class, the first alpha is alpha_0
alpha_parameters = [[1,2,2,2,2,2],
                    [1,2,2,2,2,2],
                    [1,2,2,2,2,2]]


# ================== SIMULATION PARAMETER INITIALIZATION ================== #

# Generates a consistent Item-Item influence graph given the primary_to_secondary_mapping and the lambda value 
prob_matrix = prob_matrix_generation(primary_to_secondary_mapping, lambda_param)
print('Probability matrix: \n', prob_matrix)

np.random.seed(None)

# Instantiate the bandit used to learn prices
bandit = TS_Learner(prices)
# Instantiate the estimator of the number of items sold (Step 4)
items_sold_estimator = Number_of_sold_items_estimator(5, 3)

# Setting the number of days, users and simulations to be done in the experiment
days = 30
users = 50
n_simulations = 10

# ================== OPTIMUM COMPUTATION ================== #

opt_per_starting_point = np.zeros((3,5))

# Here there is the computation to obtain the optimum value needed to compute the empirical regret of our algorithms.
# It is important to notice that the optimum is computed considering the Item-Item influence graph and the activation probabilities. In other words,
# we consider the relationships between items to optimize the learners and so we must consider this aspect also in the computation of optimum points

# Estimate activation probabilities with MonteCarlo sampling for each user class
activation_probs = []
for user_class in range(prob_matrix.shape[0]):
    estimator = MC_sampling(prob_matrix[user_class])
    activation_probs.append(estimator.estimate_activ_prob( MC_num_iterations(prob_matrix[user_class])))
print('Activation probabilities: ',activation_probs)

# Create all possible combinations of price per item to be evaluated in the optimum computation
possible_arms_indeces = np.arange(prices.shape[1])
combinations = []
for comb in itertools.product(possible_arms_indeces, repeat=len(possible_arms_indeces)+1):
  combinations.append(comb)

# Compute opts per starting point (i.e. we will obtain the optimum expected reward obtainable starting from a starting point item and then moving from it following the activation probabilities)
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

# ================== ALGORITHM SIMULATION ================== #

S = Simulator(days,
              users,
              n_simulations,
              alpha_parameters,
              seed,
              bandit,
              True, #True if the number of sold items is uncertain
              items_sold_estimator,
              True, #True if context has to be used
              False,  #True if using approximation to speed up
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


