import numpy as np

class MC_sampling:
    
    def __init__(self, prob_matrix):
        self.prob_matrix = prob_matrix
        self.node_dim = prob_matrix.shape[0]
        self.z = np.zeros(self.node_dim)
    
    #returns a vector containing activation probs of nodes
    def estimate_activ_prob(self, n_iter):
        #perform multiple iterations of MC
        for t in range(n_iter):
            #generate random live-edge graph according to prob_matrix
            live_edge_mask = np.empty(shape=[self.node_dim, self.node_dim])
            for i in range(self.node_dim):
                for j in range(self.node_dim):
                    live_edge_mask[i, j] = np.random.choice([0, 1], p=[1-self.prob_matrix[i, j], self.prob_matrix[i, j]])

            #TODO: capire questione dei seed
            seed=0
            #get all active nodes of the previously generated live-edge graph
            active_nodes = self.depth_first_tree_search(live_edge_mask, seed, visited = set())
            for i in active_nodes:
                #TODO: da capire se è giusto non considerare il seed come attivato
                if i != seed:
                    self.z[i] +=1

        return self.z / n_iter

    def depth_first_tree_search(self, live_edge_graph, start, visited = set()):
        visited.add(start)
        activated = np.where(live_edge_graph[start] == 1)[0]
        for node in activated:
            if node not in visited:
                self.depth_first_tree_search(live_edge_graph, node, visited)

        return visited


n_nodes = 5 
prob_matrix = np.random.uniform(0.0, 0.5,(n_nodes, n_nodes))
estimator = MC_sampling(prob_matrix)
activation_probs = estimator.estimate_activ_prob(100)
print(activation_probs)
