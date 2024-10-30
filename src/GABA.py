import numpy as np
from tree_utils import create_perfect_binary_tree, nodes_at_depth, find_parents
from utils import wilson_random_spanning_tree
import networkx as nx
import time
from tqdm.auto import tqdm

class GABA:
    def __init__(self, graph, K, eta = 0.01, seed = 42):
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self.K = K
        self.graph = graph
        # initialization
        self.support_tree, self.support_tree_nodes = self.constructBST(graph, seed)
        
        # NB leaves in action tree = 2**self.high_action_tree
        
        self.high_action_tree = int(np.ceil(np.log2(self.K)))
        n_nodes_action_tree = 2**(self.high_action_tree+1) - 1
        action_tree_nodes = list(range(n_nodes_action_tree))
        action_tree_nodes.reverse()
        assert len(action_tree_nodes) == n_nodes_action_tree 

        self.action_tree = create_perfect_binary_tree(action_tree_nodes)       
        
        self.mu = np.ones(self.support_tree_nodes)
        self.theta = np.zeros((self.support_tree_nodes, n_nodes_action_tree))
        
        for n in range(self.support_tree_nodes):
            for m in range(self.K):
                self.theta[n][m] = 1

            for d in range(1, self.high_action_tree + 1):
                for m in nodes_at_depth(self.action_tree, self.high_action_tree - d):
                    self.theta[n][m.value] = self.theta[n][m.left.value] + self.theta[n][m.right.value]
        
        self.xi = []
        
    def predict(self, user_node):
        # 3
        parents = find_parents(self.support_tree, user_node)
        parents_values = np.array([v.value for v in parents])
        
        # print("Parents: ", parents_values)
        # print("MU: ", self.mu[parents_values])
        # print("THETAMUL", self.theta[parents_values][:,self.action_tree.value])

        probs = self.mu[parents_values] * self.theta[parents_values][:,self.action_tree.value]
        norm = np.sum(probs)
        probs = probs / norm
        
        # print("P:", probs)
        delta_t = self.rng.choice(parents_values, p = probs)
        
        # 4
        xi_list = [self.action_tree]
        xi_prev = self.action_tree
        for d in range(self.high_action_tree):
            draw_set = [xi_prev.right, xi_prev.left]
            
            probs = np.array([self.theta[delta_t][xi_prev.right.value], self.theta[delta_t][xi_prev.left.value]])
            norm = np.sum(probs)
            probs = probs / norm
            
            xi = self.rng.choice(draw_set, p = probs)
            xi_list += [xi]
            xi_prev = xi
        # 5
        self.xi = xi_list
        return xi.value
    
    def update(self, user_node, played, loss):
        temp_mu = np.copy(self.mu)
        temp_theta = np.copy(self.theta)
        
        parents = find_parents(self.support_tree, user_node)
        parents_values = np.array([v.value for v in parents])
        psi_t = np.sum(self.mu[parents_values] * self.theta[parents_values][:,self.action_tree.value])
        rho_t = np.sum(self.mu[parents_values] * self.theta[parents_values][:,played])
        if rho_t == 0.0:
            lambda_t = 0.0
        else:
            lambda_t = np.exp(-self.eta * loss * psi_t/rho_t)
        # print("psi_t: ", psi_t)
        # print("rho_t: ", rho_t)
        # print("psi_t/rho_t: ", psi_t/rho_t)
        # print("lambda_t: ", lambda_t)
        # print("mu factor: ", psi_t / (psi_t - (1 - lambda_t)*rho_t))
        # a
        self.mu[parents_values] = self.mu[parents_values] * psi_t / (psi_t - (1 - lambda_t)*rho_t)
        self.mu[parents_values] = np.clip(self.mu[parents_values],None, 1e10)
        # np.where(np.isinf(self.mu), ,self.mu)

        for n in parents_values:
            self.theta[n][played] = lambda_t * self.theta[n][played]
            # b (skip modifications)
            # c
            for d in range(1, self.high_action_tree + 1):
                xi_g_d = self.xi[self.high_action_tree - d]
                self.theta[n][xi_g_d.value] = self.theta[n][xi_g_d.left.value] + self.theta[n][xi_g_d.right.value]
            # 9 skip modifications
                
    def constructBST(self, graph, seed = 42):
        rst = wilson_random_spanning_tree(graph, seed=seed)
        dfv = list(nx.dfs_preorder_nodes(rst))
        h = int(np.ceil(np.log2(len(dfv))))
        n_nodes = 2**(h+1) - 1
        remaining = list(range(len(dfv), n_nodes))
        #pad = [None for _ in range(n_nodes - len(dfv))]
        #n_list = dfv + pad
        result = dfv + remaining
        result.reverse()
        root = create_perfect_binary_tree(result)
        
        return root, n_nodes

def train_gaba(graph, T, K_classes, eta = None, seed = 42, verbose = True, name='GABA', loss_percentage = 1.0,
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
    if not eta:
        eta = np.sqrt(2 / (K_classes * T))
    gaba = GABA(graph, K_classes,eta = eta, seed = seed)
    # Set to draw nodes from
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
        y_hat_t = gaba.predict(x_t)
        y_hat_t_ACTION = y_hat_t - 1
        temp_pred += time.time() - temp
        
        # Verbose notifications
        # if verbose:
        #     print(f"\nTime-step {i}| node {x_t}| label: {y_t}")
        #     print(f"Predicted: {y_hat_t}")
        
        # Mistake count
        if y_hat_t_ACTION != y_t:
            tot_mistakes += 1
        results += [tot_mistakes]
        
        # loss computation
        rand = rng.random()
        loss = 0 if y_hat_t_ACTION == y_t else 1
        if rand > loss_percentage:
            loss = 0
        # Update winnow
        temp = time.time()
        gaba.update(x_t, y_hat_t, loss)
        temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), gaba
    return tot_mistakes, results