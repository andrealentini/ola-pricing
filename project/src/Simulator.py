import random

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
from parameters_generation_utils import alpha_generation

class Simulator:

    def __init__(self, days, users, alpha_parameters, seed, bandit,
                 prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr, opt):
        self.days = days
        self.users = users #should be different every day? "Every day, there is a random number of potential new customers"
        self.alpha_parameters = alpha_parameters #3x6 (3 class of users -> 3 sets of alpha)
        self.prices = prices
        self.e = Environment(prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr)
        self.seed = seed
        self.bandit = bandit
        self.n_items = self.e.prices.shape[0]
        self.opt = opt
        self.opts = []
        self.rewards = []


    def run_simulation(self, debug):

        rewards_per_day = [[] for i in range(self.n_items)]

        for day in range(0, self.days):

            alphas = alpha_generation(self.alpha_parameters, seed=self.seed)

            today_prices = self.bandit.pull_prices()

            observed_rewards = [[] for i in range(self.n_items)]

            for user in range(0, self.users):
                #retrieve the user features -> user class
                feature_1 = np.random.choice([0,1],p=[1-self.e.feature_1_dist, self.e.feature_1_dist])
                feature_2 = np.random.choice([0,1],p=[1-self.e.feature_2_dist, self.e.feature_2_dist])
                user_class = self.e.user_class_mapping(feature_1, feature_2)
                if debug: print('user class: ',user_class)

                #starting item, -1 means that the user lands on the webpage of a competitor
                items = np.concatenate((np.array([-1]), self.e.items), axis=0)
                starting_point = np.random.choice(items, p = alphas[user_class])

                #to save bought_items that we cannot visit in the future
                bought_items = np.zeros(self.n_items)

                #if the user didn't land on a competitor webpage
                if starting_point != -1:

                    primary = starting_point
                    if debug: print('starting point', primary)

                    #models the multiple paths of a user
                    items_to_visit = []

                    #exit condition: primary==1
                    while primary != -1:
                        purchase_outcome = self.e.purchase(primary, today_prices[primary], user_class)
                        if purchase_outcome:
                            if debug: print(str(primary) + ' purchased')

                            observed_rewards[primary].append(purchase_outcome)

                            n_items_sold = self.e.get_items_sold(primary, user_class)

                            self.rewards.append(self.prices[primary][today_prices[primary]] * purchase_outcome * n_items_sold)
                            self.opts.append(self.opt[user_class][primary])

                            if debug: print('items sold',n_items_sold)

                            bought_items[primary] = 1

                            #salvarsi dati per bandit del pricing

                            clicked_secondary = self.e.get_clicked_secondary(user_class, bought_items, primary)
                            if debug: print('clicked secondary',clicked_secondary)
                            items_to_visit = items_to_visit + clicked_secondary
                        else:
                            observed_rewards[primary].append(0)
                            self.rewards.append(0)
                            self.opts.append(self.opt[user_class][primary])

                        if len(items_to_visit) != 0:
                            # random.shuffle(items_to_visit) necessary?
                            primary = items_to_visit.pop()
                        else:
                            primary = -1

            self.bandit.update(today_prices, observed_rewards)

        #Days ended
        for item in range(0, self.n_items):
            rewards_per_day[item].append(self.bandit.collected_rewards_per_item[item])
    
    def plot_cumulative_regret(self):
        instant_regrets = []

        for i in range(len(self.rewards)):
            instant_regrets.append(self.opts[i] - self.rewards[i])
        
        # todo: ho considerato tutti i days come un unico episodio, non sono sicuro
        cumulative_regret = np.cumsum(instant_regrets)

        plt.plot(cumulative_regret)
        plt.show()




