import numpy as np
from src.utils import find_bases

class Winnow:
    def __init__(self, bases_list):
        self.bases_list = bases_list
        self.N = len(self.bases_list)
        self.w = np.ones(self.N, dtype=np.float32)
    
    def predict(self, node_id, node_to_base):
        '''
        Predict output based on internal state.
        '''
        active_bases_indexes = node_to_base[node_id]
        res = sum([self.w[ind] for ind in active_bases_indexes])
        res_curr = 1 if res >= self.N else 0
        return res_curr

    def update(self, node_id, predicted, true_label, node_to_base, verbose = False):
        '''
        Update winnow's weigths based on the previous prediction, check each element
        in the kernel, and scale the weigth according to the element by alpha if > 0
        '''
        active_bases_indexes = node_to_base[node_id]
        if predicted != true_label:
            for ind in active_bases_indexes:
                self.w[ind] = self.w[ind] * (2 ** (true_label - predicted))
        if verbose:
            print("Updated: ", self.w)
            
class WinnowSpaceEff:
    def __init__(self, bases_list, K, k, N, m):
        self.lmbd = k /(K*N)
        self.eta = np.sqrt(k * np.log(K * N) / m)

        self.bases_list = bases_list
        
        self.N = len(self.bases_list)
        self.w = np.full(self.N, self.lmbd)
        # self.w = [self.lmbd for _ in range(self.N)]

    def predict(self, node_id):
        '''
        Predict output based on internal state.
        '''
        active_bases_indexes = list(find_bases(self.bases_list, node_id))
        # res = sum([self.w[ind] for ind in active_bases_indexes])
        res = np.sum(self.w[active_bases_indexes])
        # res_curr = 1 if res >= 1/2 else 0
        # Leave the check to the train algorithm to brake ties correclty
        return res

    def update(self, node_id, predicted, true_label, verbose = False):
        '''
        Update winnow's weigths based on the previous prediction, check each element
        in the kernel, and scale the weigth according to the element by alpha if > 0
        '''
        active_bases_indexes = list(find_bases(self.bases_list, node_id))

        if predicted != true_label:
            if predicted == 0:
                self.w[active_bases_indexes] = self.w[active_bases_indexes] * np.exp(self.eta)
            else:
                self.w[active_bases_indexes] = self.w[active_bases_indexes] * np.exp(-self.eta)
        if verbose:
            print("Updated: ", self.w)

class WinnowStephen:
    def __init__(self, bases_list, K, k, N, m):
        self.lmbd = k /(K*N)
        self.eta = np.sqrt(k * np.log(K * N) / m)

        self.bases_list = bases_list
        
        self.N = len(self.bases_list)
        self.w = np.full(self.N, self.lmbd)
        # self.w = [self.lmbd for _ in range(self.N)]

    def predict(self, node_id, node_to_base):
        '''
        Predict output based on internal state.
        '''
        active_bases_indexes = node_to_base[node_id]
        # res = sum([self.w[ind] for ind in active_bases_indexes])
        res = np.sum(self.w[list(active_bases_indexes)])
        # res_curr = 1 if res >= 1/2 else 0
        # Leave the check to the train algorithm to brake ties correclty
        return res

    def update(self, node_id, predicted, true_label, node_to_base, verbose = False):
        '''
        Update winnow's weigths based on the previous prediction, check each element
        in the kernel, and scale the weigth according to the element by alpha if > 0
        '''
        active_bases_indexes = list(node_to_base[node_id])

        if predicted != true_label:
            if predicted == 0:
                self.w[active_bases_indexes] = self.w[active_bases_indexes] * np.exp(self.eta)
            else:
                self.w[active_bases_indexes] = self.w[active_bases_indexes] * np.exp(-self.eta)
        if verbose:
            print("Updated: ", self.w)