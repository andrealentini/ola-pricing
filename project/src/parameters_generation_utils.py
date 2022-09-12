import numpy as np
from scipy.stats import dirichlet

'''
Collection of functions that helps in the environment parameters generation
'''

#samples the alphas from a dirichlet for the 3 classes
def alpha_generation(alpha_parameters, seed, n=6):
    alphas = np.zeros((3,n))
    for user_class in range(0,3):
        alphas[user_class] = dirichlet.rvs(alpha=alpha_parameters[user_class], size=1, random_state=seed)
    return alphas

#sample a feasible probability matrix for each class
def prob_matrix_generation(primary_to_secondary_mapping, lambda_parameter):
    prob_matrix = np.zeros((3,5,5))
    for user_class in range(0,3):
        for item in range(0,5):

            first_secondary = primary_to_secondary_mapping[item][0]
            to_first_secondary_prob = np.random.uniform(0, 1)
            prob_matrix[user_class][item][first_secondary] = to_first_secondary_prob

            second_secondary = primary_to_secondary_mapping[item][1]
            to_second_secondary_prob = np.random.uniform(0, 1) * lambda_parameter
            prob_matrix[user_class][item][second_secondary] = to_second_secondary_prob

    return prob_matrix

# return the theoretical number of iterations needed to have certain confidence on the approximation result
# with probability at least 1-delta, estimated activation probs of every node is subject to an additive error +/- epsilon*n (n number of nodes)
def MC_num_iterations(prob_matrix, epsilon_n=0.05, delta=0.05):
    n = prob_matrix.shape[0]
    epsilon = epsilon_n / n
    # obs: check if this log10(n) is correct (it should be log10(|S|) where S is the set of seeds)
    return int((1/epsilon**2)*np.log10(n)*np.log10(1/delta))

