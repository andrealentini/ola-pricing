import numpy as np


class Learner:

    def __init__(self, n_arms):
        
        #n_arms is the shape of the prices: 5x4
        self.n_arms = n_arms
        
        #di solito si usa nell'aggiornamento delle witdh; qui, siccome pulliamo 5 arm e poi pesco i reward di tali arm più volte in una giornata, 
        #potremmo aggiornare t ogni volta che pesco i reward invece che ogni volta che pullo i nuovi arm (cioè a fine giornata)
        self.t = 0

        #for readability
        self.n_items=n_arms[0]
        self.n_prices=n_arms[1]

        #we have 20 arms, therefore 20 lists to keep track of the reward of a certain arm of a certain item
        #Therefore we will have a list of 5 lists representing each one an item
        self.rewards_per_item_arm= [[] for i in range(self.n_items)]
        
        #Each item will have 4 list associated, representing the prices
        for i in range(self.n_items):
            self.rewards_per_item_arm[i] = x = [[] for j in range(self.n_prices)]
        
        #The total collected rewards for each item, therefore it will be a list of 5 lists(5 items)
        self.collected_rewards_per_item = [[] for _ in range(self.n_items)]

    def update(self, pulled_arms, rewards): #pulled_arms è un vettore, mentre rewards sarà un matrice di 5x(numero di volte che pesco in una giornata)
        
        self.t += 1

        #Here i add the reward of for each arm pulled in the related item (i scorre gli item)
        for i in range(self.n_items):
            self.rewards_per_item_arm[i][pulled_arms[i]].append(rewards[i])

        #Here i add the rewards obtained from each item
        for i in range(self.n_items):
            self.collected_rewards_per_item[i].append(rewards[i]) 
