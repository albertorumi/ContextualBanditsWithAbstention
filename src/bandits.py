import numpy as np
from tqdm.auto import tqdm
import time

class OSE4:
    def __init__(self, bases_list, K, eta = 0.01, seed = 42):
        """
        bases_list : list of bases
        K : number of active actions (without abstention)
        """
        #print("bases_list: ",bases_list)
        self.N = len(bases_list)
        self.rng = np.random.default_rng(seed=seed)
        # FIXME : ASK STEPHEN for c_init value and eta
        self.c_init = 1 / self.N
        self.eta = eta
        self.bases_list = bases_list
        
        # K multiplied by the number of basis is the length of the weight vector (just one algorithm for all classes)
        self.K = K        
        # Initialize all weights to lambda value, stored in a dict form for convenience
        self.w = {
            label : np.full(self.N, self.c_init) for label in range(K)
        }
        #print(sum(self.w[0]))

    def predict(self, node_id, node_to_base):
        '''
        Predict output based on internal state.
        '''
        active_bases_indexes = list(node_to_base[node_id])
        dist_actions = np.array([np.sum(specialists[active_bases_indexes]) for _, specialists in self.w.items()])
        mu = np.sum(dist_actions)
        if mu > 1:
            # Will also return Abstention
            for label in range(self.K):
                self.w[label][active_bases_indexes] /= mu
            mu = 1    
            dist_actions = np.array([np.sum(specialists[active_bases_indexes]) for _, specialists in self.w.items()])
            # print(dist_actions)
        #rand = self.rng.random()
        #if rand <= 1 - mu:
        #    # Abstention case, noise class is indexed by -1
        #    return -1
        dist_actions = np.concatenate((np.array([1 - mu]),dist_actions))
        # if node_id == 1:
        #     print("POINT: ", node_id)
        #     print("MU", mu)
        #     print("DIST: ", dist_actions)
        res = self.rng.choice(np.arange(-1, self.K), p = dist_actions)
        return res

    def update(self, node_id, predicted, loss, node_to_base, verbose = False):
        '''
        Update based on the received loss
        '''
        if predicted == -1:
            # I've abstained so I don't get rewards or losses
            return
        active_bases_indexes = list(node_to_base[node_id])
        # c = set(np.arange(self.N))
        # not_active_indx = np.array(list(c.difference(active_bases_indexes)))
        # assert len(not_active_indx) + len(active_bases_indexes) == self.N
        sums = {k : np.sum(specialists[active_bases_indexes]) for k, specialists in self.w.items()}
        Q = set([k for k,s in sums.items() if s<self.eta])
        # for k in Q:
        #     self.w[k] = self.w[k] * np.exp(self.eta)
        
        for k in range(self.K):
            if k in Q:
                continue
            else:
                p = sums[k]
                self.w[predicted][active_bases_indexes] = self.w[predicted][active_bases_indexes]*np.exp(self.eta * (loss / p))
                

def train_bandit(graph, 
                bases_list,
                node_to_bases_ind,
                T,
                K_classes,
                eta = 0.01,
                loss_percentage = 1.0,
                seed = 42,
                verbose = True, 
                name='',
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
    ose4 = OSE4(bases_list, K_classes, eta = eta)
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
        y_hat_t = ose4.predict(x_t, node_to_bases_ind)
        temp_pred += time.time() - temp
        
        # Verbose notifications
        if verbose and y_t != y_hat_t:
            print(f"\nTime-step {i}| node {x_t}| label: {y_t}")
            print(f"Predicted: {y_hat_t}")
        
        # Mistake count
        if y_hat_t != y_t:
            tot_mistakes += 1
        results += [tot_mistakes]
        
        # loss count
        rand = rng.random()
        if y_hat_t == -1:
            # Predicted random
            reward = 0
        else:
            # Predicted something : +1 or -1 loss
            reward = 1 if y_hat_t == y_t else -1
            if rand > loss_percentage:
                # smoothing loss
                reward = 0
        # Update winnow
        temp = time.time()
        ose4.update(x_t, y_hat_t, reward, node_to_bases_ind)
        temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), ose4
    return tot_mistakes, results