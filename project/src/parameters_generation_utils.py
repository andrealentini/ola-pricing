import numpy as np
from scipy.stats import dirichlet

'''
Collection of functions that helps in the environment parameters generation
'''

#samples the alphas from a dirichlet for the 3 classes
def alpha_generation(alpha_parameters, seed):
    alphas = np.zeros((3,6))
    for user_class in range(0,3):
        alphas[user_class] = dirichlet.rvs(alpha=alpha_parameters[user_class], size=1, random_state=seed)
    return alphas

#sample a feasible probability matrix for each class
def prob_matrix_generation(primary_to_secondary_mapping, lambda_parameter):
    prob_matrix = np.zeros((3,5,5))
    for user_class in range(0,3):
        for item in range(0,5):
            primary = primary_to_secondary_mapping[item][0]
            to_primary_prob = np.random.uniform(0, 1)
            prob_matrix[user_class][item][primary] = to_primary_prob

            secondary = primary_to_secondary_mapping[item][1]
            to_secondary_prob = np.random.uniform(0, 1) * lambda_parameter
            prob_matrix[user_class][item][secondary] = to_secondary_prob

    return prob_matrix
