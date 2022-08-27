import numpy as np

class Environment:

    def __init__(self, prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr):

        self.prices = prices #the four prices for each product 5x4
        self.items = np.arange(0,5,1) #the five items
        self.prob_matrix = prob_matrix #the probability matrices 5x5x3
        self.feature_1_dist = feature_1_dist #feature 1 distribution
        self.feature_2_dist = feature_2_dist #feature 2 distribution
        self.conversion_rates = conversion_rates #5x4x3
        self.primary_to_secondary_mapping = primary_to_secondary_mapping #specify which secondary are showed after each primary
        self.n_items_to_buy_distr = n_items_to_buy_distr



    '''Computes the mapping between the user feature and the 
    corresponding class index. 
    feature_1 == 0 and feature_2 == 0 -> class 0
    feature_1 == 1 and feature_2 == 1 -> class 1
    (feature_1 == 1 and feature_2 == 0) OR (feature_1 == 0 and feature_2) == 1 -> class 2 
    '''
    def user_class_mapping(self, feature_1, feature_2):
        if feature_1 == 0 and feature_2 == 0:
            return 0
        elif feature_1 == 1 and feature_2 == 1:
            return 1
        else:
            return 2

    #equivalent of the round method
    def purchase(self, item, price, user_class, phase):
        purchase_outcome = np.random.binomial(1, self.conversion_rates[phase][user_class][item][price])
        return purchase_outcome #1 or 0 = the user buys or not

    #retrieve the secondary items clicked by the user
    def get_clicked_secondary(self, user_class, bought_items, primary):
        secondary_prob = self.prob_matrix[user_class][primary]
        clicked_secondary = []
        for secondary in self.primary_to_secondary_mapping[primary]:
            if bought_items[secondary] == 0:
                click_outcome = np.random.choice([0, 1], p=[1-secondary_prob[secondary], secondary_prob[secondary]])
                if click_outcome == 1:
                    clicked_secondary.append(secondary)
        return clicked_secondary


    def get_items_sold(self, item, user_class):
        mu, sigma = self.n_items_to_buy_distr[user_class, item]
        items_sold = int(np.round(np.random.normal(mu, sigma, 1)))
        if items_sold <= 0.0:
            items_sold = 1
        return items_sold





