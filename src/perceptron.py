import numpy as np

class Perceptron_on_graph:
    def __init__(self, L_pinv):
        '''
        Binary perceptron on the graph.
        '''
        self.L_pinv = L_pinv
        self.w = np.zeros(len(L_pinv))
    
    def predict(self, node):
        '''
        Predict the binary label (+1/-1) of the node.
        '''
        res = np.dot(self.w.T, self.L_pinv[node])
        return 1 if res >= 0 else -1
    
    def update(self, node, predicted, true_label):
        '''
        Update the weigth vector. 
        node : node to update
        predicted : predicted label (+1 or -1)
        true_label : true label (+1 or -1)
        '''
        if predicted != true_label:
            self.w[node] = self.w[node] + true_label  