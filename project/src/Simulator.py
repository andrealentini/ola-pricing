import random

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
from parameters_generation_utils import alpha_generation


class Simulator:

    def __init__(self, days, users, n_simulations, alpha_parameters, seed, bandit,
                 prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping,
                 n_items_to_buy_distr, opt_per_starting_point, activation_probs):
        self.days = days
        self.users = users  # should be different every day? "Every day, there is a random number of potential new customers"
        self.n_simulations = n_simulations
        self.alpha_parameters = alpha_parameters  # 3x6 (3 class of users -> 3 sets of alpha)
        self.prices = prices
        self.e = Environment(prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates,
                             primary_to_secondary_mapping, n_items_to_buy_distr)
        self.seed = seed
        self.bandit = bandit
        self.n_items = self.e.prices.shape[0]
        # self.opt = opt
        self.opt_per_starting_point = opt_per_starting_point
        self.activations_probs = activation_probs
        self.opts = []
        self.rewards = []
        self.R = []

    def run_simulation(self, debug):

        rewards_per_day = [[] for i in range(self.n_items)]

        for simulation in range(0, self.n_simulations):
            # print(simulation)

            # Reset the bandit at each new simulation
            self.bandit.reset()

            for day in range(0, self.days):

                alphas = alpha_generation(self.alpha_parameters, seed=self.seed)

                today_prices = self.bandit.pull_prices()

                bandit_rewards = [[] for i in range(self.n_items)]

                for user in range(0, self.users):
                    # retrieve the user features -> user class
                    feature_1 = np.random.choice([0, 1], p=[1 - self.e.feature_1_dist, self.e.feature_1_dist])
                    feature_2 = np.random.choice([0, 1], p=[1 - self.e.feature_2_dist, self.e.feature_2_dist])
                    user_class = self.e.user_class_mapping(feature_1, feature_2)
                    user_rewards = 0
                    if debug: print('user class: ', user_class)

                    # starting item, -1 means that the user lands on the webpage of a competitor
                    items = np.concatenate((np.array([-1]), self.e.items), axis=0)
                    starting_point = np.random.choice(items, p=alphas[user_class])

                    # to save bought_items that we cannot visit in the future
                    bought_items = np.zeros(self.n_items)

                    # if the user didn't land on a competitor webpage
                    if starting_point != -1:

                        primary = starting_point
                        if debug: print('starting point', primary)

                        # models the multiple paths of a user
                        items_to_visit = []

                        # exit condition: primary==1
                        while primary != -1:
                            purchase_outcome = self.e.purchase(primary, today_prices[primary], user_class)
                            if purchase_outcome:
                                if debug: print(str(primary) + ' purchased')

                                n_items_sold = self.e.get_items_sold(primary, user_class)

                                user_rewards += self.prices[primary][
                                                    today_prices[primary]] * purchase_outcome * n_items_sold

                                if debug: print('items sold', n_items_sold)

                                bought_items[primary] = 1

                                clicked_secondary = self.e.get_clicked_secondary(user_class, bought_items, primary)
                                if debug: print('clicked secondary', clicked_secondary)
                                items_to_visit = items_to_visit + clicked_secondary

                            if len(items_to_visit) != 0:
                                primary = items_to_visit.pop()
                            else:
                                primary = -1
                        bandit_rewards[starting_point].append(user_rewards)
                        self.rewards.append(user_rewards)
                        self.opts.append(self.opt_per_starting_point[user_class][starting_point])
                    else: #if starting point == -1
                        self.rewards.append(0)
                        self.opts.append(0)
                self.bandit.update(today_prices, bandit_rewards)

            # Days ended
            for item in range(0, self.n_items):
                rewards_per_day[item].append(self.bandit.collected_rewards_per_item[item])
            # Collect regrets of the simulation and reset auxiliary lists self.rewards and self.opts
            instant_regrets = []
            for i in range(len(self.rewards)):
                instant_regrets.append(self.opts[i] - self.rewards[i])
            cumulative_regret = np.cumsum(instant_regrets)
            self.R.append(cumulative_regret)
            self.rewards = []
            self.opts = []

    def plot_cumulative_regret(self):

        # Plot the mean regret within its standard deviation
        # In order to compute the mean we impose same length over all the regrets of different simulations (each simulation can have different interactions depending on users interaction)
        # min_len = min(len(i) for i in self.R)
        # for i, sublist in enumerate(self.R):
        # to_cut = len(sublist) - min_len
        # if to_cut>0:
        # self.R[i] = sublist[:-to_cut]

        mean_R = np.mean(self.R, axis=0)
        std_R = np.std(self.R, axis=0) / np.sqrt(self.n_simulations)

        plt.plot(mean_R)
        plt.fill_between(range(mean_R.shape[0]), mean_R - std_R, mean_R + std_R, alpha=0.4)
        plt.show()




