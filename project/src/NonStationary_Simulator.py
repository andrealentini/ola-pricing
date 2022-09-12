import random

import numpy as np
import matplotlib.pyplot as plt
from NonStationary_Environment import Environment
from parameters_generation_utils import alpha_generation
from UCB_Learner import UCB_Learner

from itertools import compress

class Simulator:

    def __init__(self, days, users, n_simulations, alpha_parameters, seed, bandit, items_sold_uncertain, items_sold_estimator, 
                 prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr, opt_per_starting_point, activation_probs):
        self.days = days
        self.users = users #should be different every day? "Every day, there is a random number of potential new customers"
        self.n_simulations = n_simulations
        self.alpha_parameters = alpha_parameters #3x6 (3 class of users -> 3 sets of alpha)
        self.prices = prices
        self.e = Environment(prices, prob_matrix, feature_1_dist, feature_2_dist, conversion_rates, primary_to_secondary_mapping, n_items_to_buy_distr)
        self.seed = seed
        self.bandit = bandit
        self.n_items = self.e.prices.shape[0]
        #self.opt = opt
        self.opt_per_starting_point = opt_per_starting_point
        self.activations_probs = activation_probs
        self.opts = []
        self.rewards = []
        self.R = []
        self.RW = []
        self.OPT = []
        self.items_sold_uncertain = items_sold_uncertain
        self.items_sold_estimator = items_sold_estimator
        
        self.all_rew_feat_pulledprices = []
        for i in range(3):
            self.all_rew_feat_pulledprices.append([[] for i in range(self.n_items)])
        self.max_history_memory = 10000
        self.bandit_type = type(self.bandit).__name__
        
        



    def run_simulation(self, debug):

        #rewards_per_day = [[] for i in range(self.n_items)]

        for simulation in range(0, self.n_simulations):
            print("========================")

            # when we use context information, we have a set of bandits
            self.bandit.reset()
            
            for day in range(0, self.days):
                #print(day)
                phase_length = self.days / self.e.conversion_rates.shape[0]
                phase = int(day / phase_length)

                alphas = alpha_generation(self.alpha_parameters, seed=self.seed)

                # Pull the prices for the day
                
                if self.items_sold_uncertain:
                    today_prices = self.bandit.pull_prices_activations(self.items_sold_estimator.values, self.activations_probs)
                else:
                    today_prices = self.bandit.pull_prices_activations(self.e.n_items_to_buy_distr, self.activations_probs)
                
                # Data structure to save the rewards
                bandit_rewards = [[] for i in range(self.n_items)]
                

                for user in range(0, self.users):
                    #retrieve the user features -> user class
                    feature_1 = np.random.choice([0,1],p=[1-self.e.feature_1_dist, self.e.feature_1_dist])
                    feature_2 = np.random.choice([0,1],p=[1-self.e.feature_2_dist, self.e.feature_2_dist])

                    
                    user_class = self.e.user_class_mapping(feature_1, feature_2)
                    user_rewards = 0
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
                            purchase_outcome = self.e.purchase(primary, today_prices[primary], user_class, phase)
                            if purchase_outcome:
                                if debug: print(str(primary) + ' purchased')

                                bandit_rewards[primary].append(purchase_outcome)
                                

                                n_items_sold = self.e.get_items_sold(primary, user_class)

                                if self.items_sold_uncertain:
                                    self.items_sold_estimator.update(primary, n_items_sold)

                                user_rewards += self.prices[primary][today_prices[primary]] * purchase_outcome * n_items_sold
                                #self.rewards.append(self.prices[primary][today_prices[primary]] * purchase_outcome * n_items_sold)
                                #self.opts.append(self.opt[user_class][primary])

                                if debug: print('items sold',n_items_sold)

                                bought_items[primary] = 1

                                #salvarsi dati per bandit del pricing

                                clicked_secondary = self.e.get_clicked_secondary(user_class, bought_items, primary)
                                if debug: print('clicked secondary',clicked_secondary)
                                items_to_visit = items_to_visit + clicked_secondary
                            else:
                                bandit_rewards[primary].append(0)
                                
                                #self.rewards.append(0)
                                #self.opts.append(self.opt[user_class][primary])

                            if len(items_to_visit) != 0:
                                # random.shuffle(items_to_visit) necessary?
                                primary = items_to_visit.pop()
                            else:
                                primary = -1
                        #bandit_rewards[starting_point].append(user_rewards)
                        self.rewards.append(user_rewards)
                        self.opts.append(self.opt_per_starting_point[phase][user_class][starting_point])
                        #update all the historical for context generation
                        
                    else:
                        self.rewards.append(0)
                        self.opts.append(0)
                        #update all the historical for context generation
                        
                
                # At the end of the single day, update bandits
                self.bandit.update(today_prices, bandit_rewards)
            

            #Days ended
            #for item in range(0, self.n_items):
                #rewards_per_day[item].append(self.bandit.collected_rewards_per_item[item])
            #Collect regrets of the simulation and reset auxiliary lists self.rewards and self.opts
            instant_regrets = []
            for i in range(len(self.rewards)):
                instant_regrets.append(self.opts[i] - self.rewards[i])
            cumulative_regret = np.cumsum(instant_regrets)
            self.R.append(cumulative_regret)
            self.RW.append(self.rewards)
            self.OPT.append(self.opts)
            self.rewards = []
            self.opts = []

    def plot_cumulative_regret(self):
        
        mean_R = np.mean(self.R, axis=0)
        std_R = np.std(self.R, axis=0)/np.sqrt(self.n_simulations)

        plt.plot(mean_R)
        plt.fill_between(range(mean_R.shape[0]), mean_R-std_R, mean_R+std_R, alpha=0.4)
        plt.title('Cumulative Regret', fontsize=20)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.show()

        mean_RW = np.mean(self.RW, axis=0)
        std_RW = np.std(self.RW, axis=0)/np.sqrt(self.n_simulations)
        mean_OPT = np.mean(self.OPT, axis=0)
        std_OPT = np.std(self.OPT, axis=0)/np.sqrt(self.n_simulations)

        plt.plot(mean_RW)
        plt.plot(mean_OPT, '--')
        plt.fill_between(range(mean_RW.shape[0]), mean_RW-std_RW, mean_RW+std_RW, alpha=0.4)
        plt.title('Instant Rewards', fontsize=20)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Instant Reward', fontsize=12)
        plt.legend(['SW-UCB', 'Clairvoyant'])
        plt.show()

        cum_RW = np.cumsum(self.RW, axis=1)
        cum_OPT = np.cumsum(self.OPT, axis=1)

        mean_cum_RW = np.mean(cum_RW, axis=0)
        std_cum_RW = np.std(cum_RW, axis=0)/np.sqrt(self.n_simulations)

        mean_cum_OPT = np.mean(cum_OPT, axis=0)
        std_cum_OPT = np.std(cum_OPT, axis=0)/np.sqrt(self.n_simulations)

        plt.plot(mean_cum_RW)
        plt.plot(mean_cum_OPT, '--')
        plt.fill_between(range(mean_cum_RW.shape[0]), mean_cum_RW-std_cum_RW, mean_cum_RW+std_cum_RW, alpha=0.4)
        plt.title('Cumulative Rewards', fontsize=20)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)
        plt.legend(['SW-UCB', 'Clairvoyant'])
        plt.show()
        
    
    
    
    
        

        




