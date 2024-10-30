from collections import deque, defaultdict
import random
import networkx as nx
import warnings
from src.utils import wilson_random_spanning_tree

import rst

class RST_Fabio:
    def __init__(self, graph, k, seed=42):
        '''
        Initialize the graph with the unlabeled graph. k classes (EXCLUDING NOISE, PREDICT in the interval [-1 -- K) )
        '''
        self.random_instance = random.Random(seed)
        self.random_instance.seed(seed)
        self.k = k
        self.rst_local = wilson_random_spanning_tree(graph, seed = seed)

        for n in self.rst_local.nodes:
            self.rst_local.nodes[n]['label'] = None
        self.dfv = list(nx.dfs_preorder_nodes(self.rst_local))
        
    def predict(self, node):
        '''
        Unweighted implementation of knn on graph
        '''
        # if self.rst_local.nodes[node]['label']:
        #     warnings.warn("Already discovered node...")
        #     return self.rst_local.nodes[node]['label']
        
        # Find index of the node...
        ind_l = ind_r = 0
        for i, n in enumerate(self.dfv):
            if n==node:
                ind_l = ind_r = i
                break
        
        # Do the 1-NN on the dfv line
        vote_count = defaultdict(lambda : 0)
        while True:  
            node_r = self.dfv[ind_r]
            node_l = self.dfv[ind_l]
            if self.rst_local.nodes[node_r]['label'] != None:
                vote_count[self.rst_local.nodes[node_r]['label']] += 1
            if self.rst_local.nodes[node_l]['label'] != None:
                vote_count[self.rst_local.nodes[node_l]['label']] += 1
            
            res = []
            for label, count in vote_count.items():
                if count > 0:
                    res.append(label)

            if res != []:
                rand_ind = self.random_instance.randint(0, len(res) - 1)
                return res[rand_ind]
            # Next step if no result
            ind_r += 1
            ind_l -= 1
            if ind_r >= len(self.dfv) and ind_l <0:
                random_pred = self.random_instance.randint(-1, self.k - 1)
                return random_pred
            if ind_r >= len(self.dfv):
                assert ind_l >= 0
                ind_r = len(self.dfv) - 1
            if ind_l < 0:
                assert ind_r < len(self.dfv)
                ind_l = 0
    
    def update(self, node, label):
        self.rst_local.nodes[node]['label'] = label