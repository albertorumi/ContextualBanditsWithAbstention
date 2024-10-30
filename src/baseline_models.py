from src.knn import KNN_on_graph
from src.wta import RST_Fabio
from src.mv import MV
from src.perceptron import Perceptron_on_graph
from src.utils import compute_pinv
from tqdm.auto import tqdm
import random
import numpy as np
 
def train_mv(graph, T, k, epochs = 1, verbose = True, seed = 42, disable_tqdm_train = False, disable_tqdm_epochs = False):
    '''
    Train and check the amount of errors for T rounds of k classes with k_n neighbors as votes
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    knn = MV(graph, k)
    tot_mistakes = 0
    results = []
    for _ in tqdm(range(epochs), desc="Epoch...", disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        for i in tqdm(range(T), desc=f"Training knn", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict class
            y_hat_t = knn.predict(x_t)
            
            # Verbose notifications
            if verbose:
                print(f"Time {i}| node {x_t}| label: {y_t}")
                print(f"Predicted: {y_hat_t}")
            
            # Mistake count
            if y_hat_t != y_t:
                tot_mistakes += 1
            results += [tot_mistakes]
            
            # Update winnow
            knn.update(x_t, y_t)
    return tot_mistakes, results

def train_knn(graph, T, k, k_n,epochs = 1, verbose = True, seed = 42, disable_tqdm_train = False, disable_tqdm_epochs = False):
    '''
    Train and check the amount of errors for T rounds of k classes with k_n neighbors as votes
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    knn = KNN_on_graph(graph, k, k_n)
    tot_mistakes = 0
    results = []
    for _ in tqdm(range(epochs), desc="Epoch...", disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        for i in tqdm(range(T), desc=f"Training knn", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict class
            y_hat_t = knn.predict(x_t)
            
            # Verbose notifications
            if verbose:
                print(f"Time {i}| node {x_t}| label: {y_t}")
                print(f"Predicted: {y_hat_t}")
            
            # Mistake count
            if y_hat_t != y_t:
                tot_mistakes += 1
            results += [tot_mistakes]
            
            # Update winnow
            knn.update(x_t, y_t)
    return tot_mistakes, results


def train_rst(graph, T, k, epochs = 1, verbose = True, seed = 42, disable_tqdm_epochs = False, disable_tqdm_train = False):
    '''
    Train and check the amount of errors for T rounds of k classes with RST approach
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    rst = RST_Fabio(graph, k, seed=seed)
    tot_mistakes = 0
    results = []
    for _ in tqdm(range(epochs), desc="Epoch..." , disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        for i in tqdm(range(T), desc=f"Training rst", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict class
            y_hat_t = rst.predict(x_t)
            
            # Verbose notifications
            if verbose:
                print(f"Time {i}| node {x_t}| label: {y_t}")
                print(f"Predicted: {y_hat_t}")
            
            # Mistake count
            if y_hat_t != y_t:
                tot_mistakes += 1
            results += [tot_mistakes]
            
            # Update winnow
            rst.update(x_t, y_t)
    return tot_mistakes, results

def train_perceptron(graph, T, k, epochs=1, verbose = True, seed = 42, disable_tqdm_train = False, disable_tqdm_epochs = False):
    '''
    Train the multi-class-Perceptron algorithm on the graph, for T time steps
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    tot_mistakes = 0
    results = []
    # Create K perceptron
    # v = np.array([1/len(graph.nodes) for _ in range(len(graph.nodes))])
    # b = np.outer()
    pinv = compute_pinv(graph, b=1.0, c=0.0)
    perceptrons = [Perceptron_on_graph(pinv) for _ in range(k)]
    for _ in tqdm(range(epochs), desc="Epoch...", disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        # For each trial
        for i in tqdm(range(T), desc=f"Training perceptron", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict for each winnow
            predictions = [(cls, perc.predict(x_t)) for cls, perc in enumerate(perceptrons)]
            # Get true returns
            pred_classes = list(filter(lambda x: x[1]==1, predictions))
            # Predict
            if pred_classes:
                y_hat_t = random_instance.choice(pred_classes)[0]
            else:
                y_hat_t = -1
            
            # Verbose notifications
            if verbose:
                print(f"\nTime {i}| node {x_t}| label: {y_t}")
                print(predictions)
                print(pred_classes)
                print(f"Predicted: {y_hat_t}")
            
            # Mistake count
            if y_hat_t != y_t:
                tot_mistakes += 1
            results += [tot_mistakes]
            
            # Update perceptrons
            for cls, pred in predictions:
                if cls == y_t:
                    if verbose:
                        print(f"update perceptron for class: {cls}, predicted: {pred}, true: {1}")
                    perceptrons[cls].update(x_t, pred, 1)
                else:
                    if verbose:
                        print(f"update perceptron for class: {cls}, predicted: {pred}, true: {-1}")
                    perceptrons[cls].update(x_t, pred, -1)
    return tot_mistakes, results