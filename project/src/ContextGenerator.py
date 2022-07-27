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


    def get_split(self, root, classes):
        if root.split_feature == -1:
            classes.append(root.father_features)

        print(root.__dict__)

        if root.left is not None:
            self.get_split(root.left, classes)
        if root.right is not None:
            self.get_split(root.right, classes)


    def split_feature_space(self, classes):
        father_features = np.ones(len(self.features))*-1
        root = Node(father_features, self.features, -1)
        self.feature_tree_recurrent_generation(root)
        return self.get_split(root, classes)

    def feature_tree_recurrent_generation(self, node):
        max_split_value = 0
        max_split_feature = None
        for feature in node.features_to_evaluate:
            split_value = self.evaluate_split(feature, node.father_features)
            #split_value = np.random.rand(1)
            print(split_value)
            if split_value > max_split_value:
                max_split_value = split_value
                max_split_feature = feature

        if max_split_value > 0:
            node.split_feature = max_split_feature

            features_to_evaluate = node.features_to_evaluate.copy()
            features_to_evaluate.remove(max_split_feature)

            left_child_father_features = node.father_features.copy()
            left_child_father_features[max_split_feature] = 0
            left_child = Node(left_child_father_features, features_to_evaluate, -1)

            right_child_father_features = node.father_features.copy()
            right_child_father_features[max_split_feature] = 1
            right_child = Node(right_child_father_features, features_to_evaluate, -1)

            node.left = left_child
            node.right = right_child

            self.feature_tree_recurrent_generation(left_child)
            self.feature_tree_recurrent_generation(right_child)

    # Slides notation - p_c2 = probability that context with 'feature'=1 occurs
    # c1 = context with feature = 0 / c2 = context with feature = 1
    def evaluate_split(self, feature, father_features):
        features = father_features
        features[feature] = 1
        p_c2 = self.compute_context_probability(features)
        p_c1 = 1 - p_c2
        lower_bound_c1_prob = p_c1 - np.sqrt(-(np.log(0.95) / 2 * self.feature_tuples.shape[0]))
        lower_bound_c2_prob = p_c2 - np.sqrt(-(np.log(0.95) / 2 * self.feature_tuples.shape[0]))

        rewards_c2 = self.extract_rewards(features)
        features[feature] = 0
        rewards_c1 = self.extract_rewards(features)

        lower_bound_c2_rewards = np.mean(rewards_c2) - 0.95 * np.sqrt(np.var(rewards_c2) / len(rewards_c2))
        lower_bound_c1_rewards = np.mean(rewards_c1) - 0.95 * np.sqrt(np.var(rewards_c1) / len(rewards_c1))

        features[feature] = -1
        rewards_c0 = self.extract_rewards(features)
        lower_bound_c0_rewards = np.mean(rewards_c0) - 0.95 * np.sqrt(np.var(rewards_c0) / len(rewards_c0))

        split_value = lower_bound_c1_prob * lower_bound_c1_rewards + lower_bound_c2_prob * lower_bound_c2_rewards

        if split_value > lower_bound_c0_rewards:
            return split_value
        else:
            return 0

    def extract_rewards(self, current_node):
        rewards = []
        for idx, tuple in enumerate(self.feature_tuples):
            comparison_ok = 1
            for i in range(len(tuple)):
                if current_node[i] != -1 and current_node[i] != tuple[i]:
                    comparison_ok = 0
            if comparison_ok:
                rewards.append(self.rewards[idx])
        return rewards

    def compute_context_probability(self, current_node):
        total_realizations = self.feature_tuples.shape[0]
        positive_realizations = 0
        for tuple in self.feature_tuples:
            comparison_ok = 1
            for i in range(len(tuple)):
                if current_node[i] != -1 and current_node[i] != tuple[i]:
                    comparison_ok = 0
            if comparison_ok:
                positive_realizations += 1
        return positive_realizations/total_realizations

class Node:
    def __init__(self, father_features, features_to_evaluate, split_feature):
        self.left = None
        self.right = None
        self.features_to_evaluate = features_to_evaluate
        self.split_feature = split_feature
        self.father_features = father_features #already decided features, -1 if not like (0,-1)

'''
rewards = np.array([12, 5, 20, 5, 5, 0, 0, 5, 5, 0])
feature_tuples = np.array([(1,1), (1,0), (1,0), (1,1), (1,0), (1,1), (0,1), (0,0), (1,0), (0,0)])
C = ContextGenerator()
C.update_observations(rewards, feature_tuples)
classes = []
C.split_feature_space(classes)
print(classes)
'''