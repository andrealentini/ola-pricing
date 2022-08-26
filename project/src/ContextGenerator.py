import numpy as np


class ContextGenerator:

    # the context generator is created when we already have some past observations
    def __init__(self):
        self.rewards = []
        self.feature_tuples = []
        self.features = [0,1] #vogliamo inizializzarla in modo più estendibile?
        #nel simulator le ho chiamate feature1 e feature2, forse è meglio chiamarle feature0 e feature1

    # every 2 weeks this method is called and observations are updated
    def update_observations(self, new_rewards, new_feature_tuples):
        if len(self.rewards) == 0:
            self.rewards = new_rewards
            self.feature_tuples = new_feature_tuples
        else:
            self.rewards = np.concatenate((self.rewards, new_rewards))
            self.feature_tuples = np.concatenate((self.feature_tuples, new_feature_tuples))


    def split_feature_space(self):

        first_level_split_feature = None
        #compute first-level split considering only 1 feature
        split_value_feature_0 = self.evaluate_split([[0,-1], [1,-1], [-1,-1]])
        split_value_feature_1 = self.evaluate_split([[-1,0], [-1,1], [-1,-1]])
        if split_value_feature_0 > split_value_feature_1:
            first_level_split_feature = 0
        elif split_value_feature_0 < split_value_feature_1:
            first_level_split_feature = 1
        elif (split_value_feature_0 == split_value_feature_1) and split_value_feature_0 != 0:
            first_level_split_feature = 0 #break the tie
        else: #both splits worth 0 -> no split
            return [[-1,-1]]

        splits = []

        if first_level_split_feature == 0:
            #first_level_feature = 0 - left child
            split_value = self.evaluate_split([[0, 0], [0, 1], [0,-1]])
            if split_value > 0:
                splits.append([0, 0])
                splits.append([0, 1])
            else:
                splits.append([0,-1])

            # first_level_feature = 1 - right child
            split_value = self.evaluate_split([[1, 0], [1, 1], [1,-1]])
            if split_value > 0:
                splits.append([1, 0])
                splits.append([1, 1])
            else:
                splits.append([1, -1])

        elif first_level_split_feature == 1:
            # first_level_feature = 0 - left child
            split_value = self.evaluate_split([[0, 0], [1, 0],[-1,0]])
            if split_value > 0:
                splits.append([0, 0])
                splits.append([1, 0])
            else:
                splits.append([-1, 0])

            # first_level_feature = 1 - right child
            split_value = self.evaluate_split([[0, 1], [1, 1],[-1,1]])
            if split_value > 0:
                splits.append([0, 1])
                splits.append([1, 1])
            else:
                splits.append([-1, 1])
        print("SPLITS: ", splits)
        return splits


    # Slides notation - p_c2 = probability that context with 'feature'=1 occurs
    # c1 = context with feature = 0 / c2 = context with feature = 1
    def evaluate_split(self, split):

        rewards_c2 = self.extract_rewards(split[1])
        rewards_c1 = self.extract_rewards(split[0])
        rewards_c0 = self.extract_rewards(split[2])

        p_c2 = self.compute_context_probability(split[1])
        p_c1 = 1 - p_c2

        lower_bound_c2_rewards = np.mean(rewards_c2) - np.sqrt(-((np.log(0.90)) / (2 * len(rewards_c0))))
        lower_bound_c1_rewards = np.mean(rewards_c1) - np.sqrt(-((np.log(0.90)) / (2 * len(rewards_c0))))


        lower_bound_c0_rewards = np.mean(rewards_c0) - np.sqrt(-((np.log(0.90)) / (2 * len(rewards_c0))))

        #split_value = lower_bound_c1_prob * lower_bound_c1_rewards + lower_bound_c2_prob * lower_bound_c2_rewards
        #evaluate the split value
        split_value = p_c1 * lower_bound_c1_rewards + p_c2 * lower_bound_c2_rewards

        if split_value > lower_bound_c0_rewards:
            #print("SPLIT DONE")
            return split_value
        else:
            return 0

    def extract_rewards(self, split):
        rewards = []
        for idx, realization in enumerate(self.feature_tuples):
            comparison_ok = 1
            for i in range(len(realization)):
                if split[i] != -1 and split[i] != realization[i]:
                    comparison_ok = 0
            if comparison_ok:
                rewards.append(self.rewards[idx])
        return rewards

    def compute_context_probability(self, split):
        total_realizations = self.feature_tuples.shape[0]
        positive_realizations = 0
        for realization in self.feature_tuples:
            comparison_ok = 1
            for i in range(len(realization)):
                if split[i] != -1 and split[i] != realization[i]:
                    comparison_ok = 0
            if comparison_ok:
                positive_realizations += 1
        return positive_realizations/total_realizations
