from collections import defaultdict

import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import time

from src.exp3 import EXP3

class ContSimInfo:
    def __init__(self, graph, distance, K, eta_exp3 = None, c_y = 10, d_y = 5,seed = 42):
        """
        bases_list : list of bases
        K : number of active actions
        """
        # WEIRD PARAMETERS
        self.c_y = c_y
        self.d_y = d_y
        # STANDARD
        self.K = K
        self.graph = graph
        self.distance = distance
        self.rng = np.random.default_rng(seed=seed)
        self.EXP3_dict = dict()
        self.eta_exp3 = eta_exp3

        # init
        start_node = self.rng.integers(low=0, high =len(graph.nodes))
        B = self.find_ball(start_node, 1)
        self.A = [(B, 1)]
        self.A_star = [(B, 1)]
        
        self.count = defaultdict(lambda : 0)
        self.EXP3_dict[B] = EXP3(K, T=1, eta = self.eta_exp3)
        
    def predict(self, node_id):
        '''
        Predict output based on internal state.
        '''
        B = None
        for a,r in self.A_star:
            if node_id in a:
                B = a
                self.B = B
                self.B_radius = r
                break
        if not B:
            for a,r in self.A:
                if node_id in a:
                    r_next = r/2
                    B = self.find_ball(node_id, r_next, self.distance)
                    self.A += [(B, r_next)]
                    self.A_star += [(B, r_next)]
                    self.B = B
                    self.B_radius = r_next
                    self.EXP3_dict[B] = EXP3(self.K, T=self.T_0(r_next) + 1, eta=self.eta_exp3)
        assert B is not None, "ERROR in retrieving base"
        
        action = self.EXP3_dict[self.B].predict()
        return action
        

    def update(self, node_id, predicted, loss, verbose = False):
        self.EXP3_dict[self.B].update(predicted, loss)
        self.count[self.B] += 1
        if self.count[self.B] == self.T_0(self.B_radius):
            self.A_star.remove(self.B)

    def find_ball(self, node, radius):
        res = set()
        for n in self.graph.nodes:
            dist = self.distance(node, n)
            if dist <= radius:
                res.add(n)
        return frozenset(res)
    
    def T_0(self, radius):
        return self.c_y * radius **(-(2 + self.d_y)) * np.log(1/radius)

class Distances:
    def __init__(self, graph):
        self.graph = graph
        self.diameter = nx.diameter(graph)
        
    def custom_sp_dist(self, x,y):
        """
        Have to return a distance in a normalized format (all distances <= 1)
        """
        return nx.shortest_path_length(self.graph, source = x, target = y)/self.diameter



# def train_cont(testg, K_classes):
#     #testg = create_multi_class_clique(n_clique=300,classes=4, n_nodes_noise=600, rand_inst=rand_inst, shading = 0.3)
#     dist = Distances(testg)
#     learner = ContSimInfo(testg, dist.custom_sp_dist, K_classes + 1)

#     res = 0
#     rr = []
#     for n in tqdm(range(len(testg))):
#         action = learner.predict(n)
#         action_rr = action - 1
#         loss = 0 if action_rr == testg.nodes[n]['label'] else 1
        
#         res += loss
#         rr += [res]
        
#         learner.update(n, action, loss)
#     return res, rr

def train_cont(graph, T, K_classes, c_y = 10, d_y = 5, eta_exp3 = None, seed = 42, verbose = True, name='ContSim', loss_percentage = 1.0,
                disable_tqdm_train = False, debug = False):
    '''
    Train the multi-class-winnow algorithm on the graph, for T time steps, with given bases list
    '''
    temp_pred = 0
    temp_upd = 0
    tot_mistakes = 0
    results = []
    rng = np.random.default_rng(seed = seed)

    # Initialize algorithm
    dist = Distances(graph)
    learner = ContSimInfo(graph, dist.custom_sp_dist, K_classes, eta_exp3=eta_exp3)
    
    unlabeled_nodes = set(graph.nodes)
    # For each trial
    for i in tqdm(range(T), desc=f"Training {name}", disable=disable_tqdm_train):
        # if not unlabeled_nodes:
        #     break
        # Draw an unlabeled node and its label
        x_t = rng.choice(list(unlabeled_nodes))
        # unlabeled_nodes.remove(x_t)
        y_t = graph.nodes[x_t]['label']
        
        # Predict
        temp = time.time()
        y_hat_t = learner.predict(x_t)
        y_hat_t_ACTION = y_hat_t - 1
        temp_pred += time.time() - temp

        # loss computation
        rand = rng.random()
        loss = 0 if y_hat_t_ACTION == y_t else 1
        if rand > loss_percentage:
            loss = 0

        tot_mistakes += int(y_hat_t_ACTION != y_t)
        results += [tot_mistakes]
        # Update winnow
        temp = time.time()
        learner.update(x_t, y_hat_t, loss)
        temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), learner
    return tot_mistakes, results