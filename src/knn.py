from collections import deque, defaultdict
import random
import warnings
class KNN_on_graph:
    def __init__(self, graph, k, k_n, seed = 42):
        '''
        Initialize the graph with the unlabeled graph. k is the number of classes, k_n is the number of neighbords
        '''
        
        self.random_instance = random.Random(seed)
        self.random_instance.seed(seed)
        self.graph = graph.copy()
        for n in self.graph.nodes:
            self.graph.nodes[n]['label'] = None
        self.k = k
        self.k_n = k_n

    def predict(self, node):
        '''
        Unweighted implementation of knn on graph
        '''
        if self.graph.nodes[node]['label'] != None:
        #    warnings.warn("Already discovered node...")
           return self.graph.nodes[node]['label']
        k_n = self.k_n
        vote_count = defaultdict(lambda : 0)
        nodes_to_check = deque()
        checked = set([node])
        for n in self.graph[node]:
            nodes_to_check.append(n)
            checked.add(n)
        while k_n > 0:
            # Retrieve the node from the queue if it has neighbors available
            try:
                curr_node = nodes_to_check.popleft()
            except:
                # warnings.warn(f"Using less neighbors for node {node}, tot used: {self.k_n - k_n}")
                if vote_count:
                    max_key = max(vote_count, key=lambda k: vote_count[k])
                    # print(vote_count)
                    return max_key
                else:
                    random_pred = self.random_instance.randint(0, self.k)
                    return random_pred
            # Increase the counter
            if self.graph.nodes[curr_node]['label'] != None:
                vote_count[self.graph.nodes[curr_node]['label']] += 1
                k_n -= 1
            # Add nodes to check
            for n in self.graph[curr_node]:
                if n not in checked:
                    nodes_to_check.append(n)
                    checked.add(n)
        # Return max
        if vote_count:
            max_key = max(vote_count, key=lambda k: vote_count[k])
            # print(vote_count)
            return max_key
        random_pred = self.random_instance.randint(0, self.k)
        return random_pred
    
    def update(self, node, label):
        self.graph.nodes[node]['label'] = label