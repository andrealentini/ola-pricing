import random

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
from parameters_generation_utils import alpha_generation
from ContextGenerator import ContextGenerator
from UCB_Learner import UCB_Learner
from TS_Learner import TS_Learner
from itertools import compress

class Simulator:

    def __init__(self, days, users, n_simulations, alpha_parameters, seed, bandit, items_sold_uncertain, items_sold_estimator, use_context,
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
        self.items_sold_uncertain = items_sold_uncertain
        self.items_sold_estimator = items_sold_estimator
        #attribute for context handling
        self.use_context = use_context
        self.context_rewards = []
        self.context_feature_tuples = []
        self.all_rew_feat_pulledprices = []
        for i in range(3):
            self.all_rew_feat_pulledprices.append([[] for i in range(self.n_items)])
        self.max_history_memory = 10000
        self.bandit_type = type(self.bandit).__name__
        self.context_bandits = []
        self.context_generator = ContextGenerator()



    def run_simulation(self, debug):

        #rewards_per_day = [[] for i in range(self.n_items)]

        for simulation in range(0, self.n_simulations):

            self.context_generator = ContextGenerator()

            # when we use context information, we have a set of bandits
            if self.use_context:
                self.context_bandits = []
                self.context_bandits.append(self.create_context_bandit(np.array([-1, -1])))
            
            #Reset the bandit at each new simulation
            if not self.use_context:
                self.bandit.reset()
            else:
                for bandit in self.context_bandits:
                    bandit.reset()
            

            for day in range(0, self.days):
                #print(day)

                alphas = alpha_generation(self.alpha_parameters, seed=self.seed)

                # when context info is used, every 2 weeks we create new bandits for each feature split
                if self.use_context and day%14 == 0 and day!=0:
                    self.context_bandits = []
                    #update the context generator algorithm
                    self.context_generator.update_observations(np.array(self.context_rewards), np.array(self.context_feature_tuples))
                    self.context_rewards = []
                    self.context_feature_tuples = []
                    #split the feature space with the updated informations
                    splits = self.context_generator.split_feature_space()
                    #create a bandit for each split
                    for split in splits:
                        self.context_bandits.append(self.create_context_bandit(split))
                    # give to new bandits all and only previous observations related to their split
                    for bandit_idx, bandit in enumerate(self.context_bandits):
                        split_bandit_rewards = [[] for i in range(self.n_items)]
                        split_pulled_prices = [[] for i in range(self.n_items)]
                        all_pulled_in_split = []
                        # filter by current bandit split
                        for item in range(self.n_items):
                            rewards = self.all_rew_feat_pulledprices[0][item]
                            feats = self.all_rew_feat_pulledprices[1][item]
                            pulled_prices = self.all_rew_feat_pulledprices[2][item]
                            context_mask = []
                            for i in range(len(rewards)):
                                context_mask.append(self.get_split_idx(feats[i][0], feats[i][1]) == bandit_idx)
                            split_bandit_rewards[item] = list(compress(rewards, context_mask))
                            split_pulled_prices[item] = list(compress(pulled_prices, context_mask))
                            for prices in split_pulled_prices[item]:
                                if prices not in all_pulled_in_split:
                                    all_pulled_in_split.append(prices)
                        # filter by daily rewards with different pulled arms
                        for prices in all_pulled_in_split:
                            split_daily_bandit_rewards = [[] for i in range(self.n_items)]
                            for item in range(self.n_items):
                                daily_mask = []
                                for i  in range(len(split_bandit_rewards[item])):
                                    daily_mask.append(prices == split_pulled_prices[item][i])
                                split_daily_bandit_rewards[item] = list(compress(split_bandit_rewards[item], daily_mask))
                            bandit.update(prices, split_daily_bandit_rewards)

                        
                        #bandit.update(, split_bandit_rewards)

                # Pull the prices for the day
                if not self.use_context:
                    if self.items_sold_uncertain:
                        today_prices = self.bandit.pull_prices_activations(self.items_sold_estimator.values, self.activations_probs)
                    else:
                        today_prices = self.bandit.pull_prices_activations(self.e.n_items_to_buy_distr, self.activations_probs)
                else:
                    context_today_prices = []
                    if self.items_sold_uncertain:
                        for bandit in self.context_bandits:
                            context_today_prices.append(bandit.pull_prices_activations(self.items_sold_estimator.values, self.activations_probs))
                    else:
                        for bandit in self.context_bandits:
                            context_today_prices.append(bandit.pull_prices_activations(self.e.n_items_to_buy_distr, self.activations_probs))
                
                # Data structure to save the rewards
                if not self.use_context:
                    bandit_rewards = [[] for i in range(self.n_items)]
                else:
                    context_bandit_rewards = []
                    for bandit in range(len(self.context_bandits)):
                        context_bandit_rewards.append([[] for i in range(self.n_items)])

                for user in range(0, self.users):
                    #retrieve the user features -> user class
                    feature_1 = np.random.choice([0,1],p=[1-self.e.feature_1_dist, self.e.feature_1_dist])
                    feature_2 = np.random.choice([0,1],p=[1-self.e.feature_2_dist, self.e.feature_2_dist])

                    #if context is used we need to select the correct pulled prices
                    if self.use_context:
                        today_prices = context_today_prices[self.get_split_idx(feature_1, feature_2)]
                        bandit_rewards = context_bandit_rewards[self.get_split_idx(feature_1, feature_2)]

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
                            purchase_outcome = self.e.purchase(primary, today_prices[primary], user_class)
                            if purchase_outcome:
                                if debug: print(str(primary) + ' purchased')

                                bandit_rewards[primary].append(purchase_outcome)
                                if self.use_context:
                                    if len(self.all_rew_feat_pulledprices[0][primary]) >= self.max_history_memory:
                                        del self.all_rew_feat_pulledprices[0][primary][0]
                                        del self.all_rew_feat_pulledprices[1][primary][0]
                                        del self.all_rew_feat_pulledprices[2][primary][0]
                                    self.all_rew_feat_pulledprices[0][primary].append(purchase_outcome)
                                    self.all_rew_feat_pulledprices[1][primary].append((feature_1, feature_2))
                                    self.all_rew_feat_pulledprices[2][primary].append(today_prices)

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
                                if self.use_context:
                                    if len(self.all_rew_feat_pulledprices[0][primary]) >= self.max_history_memory:
                                        del self.all_rew_feat_pulledprices[0][primary][0]
                                        del self.all_rew_feat_pulledprices[1][primary][0]
                                        del self.all_rew_feat_pulledprices[2][primary][0]
                                    self.all_rew_feat_pulledprices[0][primary].append(0)
                                    self.all_rew_feat_pulledprices[1][primary].append((feature_1, feature_2))
                                    self.all_rew_feat_pulledprices[2][primary].append(today_prices)
                                #self.rewards.append(0)
                                #self.opts.append(self.opt[user_class][primary])

                            if len(items_to_visit) != 0:
                                # random.shuffle(items_to_visit) necessary?
                                primary = items_to_visit.pop()
                            else:
                                primary = -1
                        #bandit_rewards[starting_point].append(user_rewards)
                        self.rewards.append(user_rewards)
                        self.opts.append(self.opt_per_starting_point[user_class][starting_point])
                        #update all the historical for context generation
                        if self.use_context:
                            self.context_rewards.append(user_rewards)
                            self.context_feature_tuples.append((feature_1, feature_2))
                    else:
                        self.rewards.append(0)
                        self.opts.append(0)
                        #update all the historical for context generation
                        if self.use_context:
                            self.context_rewards.append(0)
                            self.context_feature_tuples.append((feature_1, feature_2))
                
                # At the end of the single day, update bandits
                if not self.use_context:
                    self.bandit.update(today_prices, bandit_rewards)
                else:
                    for split in range(len(self.context_bandits)):
                        self.context_bandits[split].update(context_today_prices[split], context_bandit_rewards[split])

            #Days ended
            #for item in range(0, self.n_items):
                #rewards_per_day[item].append(self.bandit.collected_rewards_per_item[item])
            #Collect regrets of the simulation and reset auxiliary lists self.rewards and self.opts
            instant_regrets = []
            for i in range(len(self.rewards)):
                instant_regrets.append(self.opts[i] - self.rewards[i])
            cumulative_regret = np.cumsum(instant_regrets)
            self.R.append(cumulative_regret)
            self.rewards = []
            self.opts = []

    def plot_cumulative_regret(self):
        
        #Plot the mean regret within its standard deviation
        # In order to compute the mean we impose same length over all the regrets of different simulations (each simulation can have different interactions depending on users interaction)
        #min_len = min(len(i) for i in self.R)
        #for i, sublist in enumerate(self.R):
            #to_cut = len(sublist) - min_len
            #if to_cut>0:
                #self.R[i] = sublist[:-to_cut]
        
        mean_R = np.mean(self.R, axis=0)
        std_R = np.std(self.R, axis=0)/np.sqrt(self.n_simulations)

        plt.plot(mean_R)
        plt.fill_between(range(mean_R.shape[0]), mean_R-std_R, mean_R+std_R, alpha=0.4)
        plt.show()
    
    def create_context_bandit(self, split):
        if self.bandit_type == 'UCB_Learner':
            new_bandit = UCB_Learner(self.prices, split)
        elif self.bandit_type == 'TS_Learner':
            new_bandit = TS_Learner(self.prices, split)
        else:
            print("not valid learner selected")
        return new_bandit
    
    def get_split_idx(self, feature_1, feature_2):
        splits = []
        for bandit in self.context_bandits:
            splits.append(bandit.get_bandit_split())
        f1_ok = False
        f2_ok = False
        for idx, split in enumerate(splits):
            if split[0] == feature_1:
                f1_ok = True
            if split[1] == feature_2:
                f2_ok = True
            if split[0] == -1:
                f1_ok = True
            if split[1] == -1:
                f2_ok = True

            if f1_ok==True and f2_ok==True:
                return idx
            else:
                f1_ok = False
                f2_ok = False

        

        




