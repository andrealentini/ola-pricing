import numpy as np

# This is a general learner class from which all other learners are derived. In our project, a learner directly works in parallel over all the items of the problem,
# instead of using more learners for different items
# prices : every learner receive as input the entire set of possible prices in a matrix that for each item (rows) there are some candidate prices (columns)
# bandit_split : optional argument that assign a contextual split of the feature space to the learner (only used in Step 7 to work with context)
class Learner:
    
    # Learner initialization
    def __init__(self, prices, bandit_split=None):
        
        self.t = 0

        self.n_items = prices.shape[0]
        self.n_prices = prices.shape[1]

        # When context is used, this attribute represent the split handled by the bandit
        self.bandit_split = bandit_split

        # List of lists containing for each item a list containing for each arm, all the rewards obtained by that arm
        self.rewards_per_item_arm = [[] for i in range(self.n_items)]
        for i in range(self.n_items):
            self.rewards_per_item_arm[i] = x = [[] for j in range(self.n_prices)]
        
        # The total collected rewards for each item, not considering the arm from which the reward is collected
        self.collected_rewards_per_item = [[] for _ in range(self.n_items)]
    
    # Method implemented by subclasses that completely reset the learner progresses, called at the end of a simulation
    def reset(self):
        pass

    # General method used to update learner parameters after the observation of a series of rewards
    # pulled_arms : array containing the prices (arms) selected to collect the rewards
    # rewards : collection of empirical rewards represented as a list of lists containing all the rewards for each item
    def update(self, pulled_arms, rewards):
        
        self.t += 1

        # Update the reward of for each arm pulled in the related item
        for i in range(self.n_items):
            self.rewards_per_item_arm[i][pulled_arms[i]] = self.rewards_per_item_arm[i][pulled_arms[i]] + rewards[i]

        # Update the rewards obtained from each item
        for i in range(self.n_items):
            self.collected_rewards_per_item[i] = self.collected_rewards_per_item[i] + rewards[i]
    
    # Simple getter used to know what is the contextual split of the learner (None if no specific split is assigned)
    def get_bandit_split(self):
        return self.bandit_split
