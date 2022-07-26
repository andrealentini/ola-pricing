import numpy as np

class ContextGenerator:

    # the context generator is created when we already have some past observations
    def __init__(self, rewards, feature_tuples):
        self.rewards = rewards
        self.feature_tuples = feature_tuples
    
    # every 2 weeks this method is called and observations are updated
    def update_observations(self, new_rewards, new_feature_tuples):
        self.rewards += new_rewards
        new_feature_tuples += new_feature_tuples

    def split_feature_space(self,):
        #to be implemented
        # greedy algorithm that split the feature space
        pass

    def user_class_mapping(self, feature1, feature2):
        #to be implemented
        # to be called after the split, given the value of the two features it returns the corresponding class
        pass