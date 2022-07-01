import random

import numpy as np
from Environment import Environment


class Simulator:

    def __init__(self, days, users,
                 prices, prob_matrix, alphas, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr):
        self.days = days
        self.users = users #should be different every day? "Every day, there is a random number of potential new customers"
        #self.bandit = bandit
        self.e = Environment(prices, prob_matrix, alphas, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr)

    def run_simulation(self, debug):

        for day in range(0, self.days):

            #today_prices = bandit.pull_prices() at the beginning of a new day we have to set the prices
            today_prices = [0,0,0,0,0] #index of the price of each item

            for user in range(0, self.users):
                #retrieve the user features -> user class
                feature_1 = np.random.choice([0,1],p=[1-self.e.feature_1_dist, self.e.feature_1_dist])
                feature_2 = np.random.choice([0,1],p=[1-self.e.feature_2_dist, self.e.feature_2_dist])
                user_class = self.e.user_class_mapping(feature_1, feature_2)
                if debug: print('user class: ',user_class)

                #starting item, -1 means that the user lands on the webpage of a competitor
                items = np.concatenate((np.array([-1]), self.e.items), axis=0)
                starting_point = np.random.choice(items, p=self.e.alphas[user_class])

                #to save bought_items that we cannot visit in the future
                bought_items = np.zeros(self.e.items.shape[0])

                #if the user didn't land on a competitor webpage
                if starting_point != -1:

                    primary = starting_point
                    if debug: print('starting point', primary)

                    #models the multiple paths of a user
                    items_to_visit = []

                    #exit condition: primary==1
                    while primary != -1:
                        if self.e.purchase(primary, today_prices[primary], user_class):
                            if debug: print(str(primary) + ' purchased')

                            n_items_sold = self.e.get_items_sold(primary, user_class)
                            if debug: print('items sold',n_items_sold)

                            bought_items[primary] = 1

                            #salvarsi dati per bandit del pricing

                            clicked_secondary = self.e.get_clicked_secondary(user_class, bought_items, primary)
                            if debug: print('clicked secondary',clicked_secondary)
                            items_to_visit = items_to_visit + clicked_secondary

                            #retrieve the current primary
                            #random.shuffle(items_to_visit) necessary?

                        if len(items_to_visit) != 0:
                            primary = items_to_visit[-1]
                            items_to_visit.pop()
                        else:
                            primary = -1

            #GIORNATA FINITA
            #aggiornare bandit pricing


