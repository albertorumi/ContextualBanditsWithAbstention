import random
from collections import defaultdict

class MV:
    def __init__(self, graph, k, seed = 42):
        '''
        Initialize the graph with the unlabeled graph. 
        k is the number of FOREGROUND classes
        '''
        self.random_instance = random.Random(seed)
        self.random_instance.seed(seed)
        self.graph = graph.copy()
        for n in self.graph.nodes:
            self.graph.nodes[n]['label'] = None
        self.k = k

    def predict(self, node):
        '''
        Unweighted implementation of knn on graph
        '''
        if self.graph.nodes[node]['label'] != None:
            return self.graph.nodes[node]['label']
        
        vote_count = defaultdict(lambda : 0)
        for n in self.graph[node]:
            if self.graph.nodes[n]['label'] != None:
                vote_count[self.graph.nodes[n]['label']] += 1
        
        # Return max
        
        if vote_count:
            max_val = max(vote_count.values())
            possible_labs = list(filter(lambda x : x[1] == max_val, list(vote_count.items())))
            return random.choice(possible_labs)[0]
        
        random_pred = self.random_instance.randint(-1, self.k)
        return random_pred
        
    def update(self, node, label):
        self.graph.nodes[node]['label'] = label