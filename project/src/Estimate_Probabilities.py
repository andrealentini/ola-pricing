import numpy as np
from copy import copy
from threading import Thread, Lock

class Probabilities_Estimator:

    def __init__(self, init_prob_matrix, alpha_parameters, items):
        self.init_prob_matrix = init_prob_matrix
        self.alpha_parameters = alpha_parameters
        self.items = items

    def simulate_graph_episode(self, n_steps_max):
        prob_matrix = self.init_prob_matrix
        
        # user classes are aggregated by mean
        aggregated_alphas = np.mean(self.alpha_parameters, axis=0)
        prob_matrix = np.mean(self.init_prob_matrix, axis=0)

        # after aggregation alphas could not sum to 1, so normalize
        aggregated_alphas = aggregated_alphas / sum(aggregated_alphas)
        
        # initial active node generated using alphas
        initial_active_nodes = np.zeros(len(self.items))
        initial_active_nodes[np.random.choice(self.items, p = aggregated_alphas)] = 1

        history = np.array([initial_active_nodes])
        active_nodes = initial_active_nodes
        newly_active_nodes = active_nodes
        t = 0
        while t<n_steps_max and np.sum(newly_active_nodes)>0:
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            prob_matrix = prob_matrix * ((p!=0) == activated_edges)
            newly_active_nodes = (np.sum(activated_edges, axis=0)>0) * (1-active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)
            history = np.concatenate((history, [newly_active_nodes]), axis = 0)
            t += 1
        return history

    def estimate_node_probabilities(self, dataset, node_index):
        n_nodes = len(self.items)
        estimated_prob = np.ones(n_nodes)*1.0/(n_nodes - 1)
        credits = np.zeros(n_nodes)
        occur_v_active = np.zeros(n_nodes)
        for episode in dataset:
            idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
            if len(idx_w_active)>0 and idx_w_active>0:
                active_nodes_in_prev_step = episode[idx_w_active - 1,:].reshape(-1)
                credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)
            for v in range(0, n_nodes):
                if(v!=node_index):
                    idx_v_active = np.argwhere(episode[:, v]==1).reshape(-1)
                    if len(idx_v_active)>0 and (idx_v_active<idx_w_active or len(idx_w_active)==0):
                        occur_v_active[v] += 1
        estimated_prob = credits/occur_v_active
        estimated_prob = np.nan_to_num(estimated_prob)
        return estimated_prob
    
    def estimate_probabilities(self, n_episodes=1000, n_steps_max=10):
        dataset = []

        for e in range(0, n_episodes):
            episode = self.simulate_graph_episode(n_steps_max)
            dataset.append(episode)
        
        estimated_prob_matrix = []
        n_nodes = len(self.items)
        for node_index in range(n_nodes):
            estimate = self.estimate_node_probabilities(dataset, node_index).tolist()
            estimated_prob_matrix.append(estimate)

        return np.array(estimated_prob_matrix).T