import numpy as np
import time
from tqdm.auto import tqdm

class EXP3:
    def __init__(self, K, T = None, eta = 0.01, seed = 42):
        self.rng = np.random.default_rng(seed)
        self.K = K
        if T != None:
            eta = np.sqrt(K/T)
        self.eta = eta
        self.p = np.array([1/K for _ in range(K)])
        
    def predict(self):
        '''
        Sample an arms according to distribution
        '''
        action = self.rng.choice(np.arange(self.K), p = self.p)
        return action
    
    def update(self, played, loss, verbose = False):
        '''
        Update distribution based on the reward received.
        inputs: 
            int : ind of played arm
            int : binary reward
        '''
        estimates = np.ones(self.K)
        if self.p[played] > 0:
            estimates[played] = np.exp(-self.eta * (loss / self.p[played]))
        p = self.p * estimates
        fac = sum(p)
        self.p = p / fac

def train_exp3(graph, T, K_classes, eta = 0.01, seed = 42, verbose = True, name='EXP3', loss_percentage = 1.0,
                disable_tqdm_train = False, debug = False):
    '''
    Train the multi-class-winnow algorithm on the graph, for T time steps, with given bases list
    '''
    temp_pred = 0
    temp_upd = 0
    tot_mistakes = 0
    results = []
    rng = np.random.default_rng(seed = seed)

    exp3_dict = {
        n : EXP3(K_classes, eta = eta, seed=seed) for n in graph.nodes
    }
    res = []
    unlabeled_nodes = set(graph.nodes)
    # For each trial
    for i in tqdm(range(T), desc=f"Training {name}", disable=disable_tqdm_train):
        # Draw an unlabeled node and its label
        x_t = rng.choice(list(unlabeled_nodes))
        y_t = graph.nodes[x_t]['label']
        
        # Predict
        temp = time.time()
        y_hat_t = exp3_dict[x_t].predict()
        y_hat_t_ACTION = y_hat_t - 1
        res += [y_hat_t_ACTION]
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
        exp3_dict[x_t].update(y_hat_t, loss)
        temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), exp3_dict
    return tot_mistakes, results, res