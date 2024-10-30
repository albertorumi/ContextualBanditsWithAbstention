import random
import time
import numpy as np
import networkx as nx
import sys
import math
from tqdm.auto import tqdm
sys.path.append("../../convgraph")
sys.path.append("../../convgraph/src")
from winnow import Winnow, WinnowSpaceEff, WinnowStephen
from utils import compute_node_to_bases, find_bases
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from itertools import compress

def retrieve_interval_bases(graph, disable = False):
    bases = set()
    dist = nx.floyd_warshall_numpy(graph)
    n = graph.number_of_nodes()
    node_list = list(graph.nodes())
    for s in range(n):
        for t in range(n):
            if s <= t:
                bases.add(frozenset(list(compress(node_list,dist[s,:]+dist[t,:]==dist[s,t]))))
    bases_list = list(bases)
    node_to_bases = compute_node_to_bases(graph, bases_list, disable=disable)
    return bases_list, node_to_bases

def retrieve_st_bases(graph):
    assert False, "Deprecated"
    bases = set()
    for s in graph.nodes:
        for t in graph.nodes:
            if s != t:
                path = frozenset(nx.shortest_path(graph,s,t))
                bases.add(path)
    return list(bases)

def retrieve_lvc_bases(graph, disable = False):
    bases_list = nx.community.louvain_communities(graph)
    node_to_bases = compute_node_to_bases(graph, bases_list, disable=disable)

    return bases_list, node_to_bases

def retrieve_lvc_bases_peeling(graph, disable=False):
    lvc = nx.community.louvain_communities(graph)
    # Peeling
    res = set()
    for n in graph.nodes:
        res.add(frozenset([n]))
    for base in lvc:
        subg = nx.subgraph(graph, base)
        node_degree = subg.degree()
        res.add(frozenset(base))
        degrees = sorted([(node_degree[n], n) for n in base], key=lambda x: x[0], reverse = False)
        for _,node in degrees:
            base.remove(node)
            if base:
                res.add(frozenset(base))
    
    bases_list = list(res)
    node_to_bases = compute_node_to_bases(graph, bases_list, disable=disable)

    return bases_list, node_to_bases

def extract_levels(row_clusters, labels):
    clusters = {}
    for row in range(row_clusters.shape[0]):
        cluster_n = row + len(labels)
        # which clusters / labels are present in this row
        glob1, glob2 = row_clusters[row, 0], row_clusters[row, 1]

        # if this is a cluster, pull the cluster
        this_clust = []
        for glob in [glob1, glob2]:
            if glob > (len(labels)-1):
                this_clust += clusters[glob]
            # if it isn't, add the label to this cluster
            else:
                this_clust.append(glob)

        clusters[cluster_n] = this_clust
    res = set()
    for v in clusters.values():
        res.add(frozenset(v))
    return res

def retrieve_hier_bases(graph, plot_dendrogram = False, disable=False):
    R = nx.floyd_warshall_numpy(graph)
    RR = squareform(R)
    Z = linkage(RR, method='ward')
    dendrogram_dict = dendrogram(Z, no_plot=not plot_dendrogram)
    
    
    res = extract_levels(Z, list(range(len(graph.nodes))))
    for n in graph.nodes:
        res.add(frozenset([n]))
    bases_list = list(res)
    node_to_bases = compute_node_to_bases(graph, bases_list, disable=disable)

    return bases_list, node_to_bases

# def retrieve_interval_bases(graph, disable = False):
#     bases = set()
#     for s in graph.nodes:
#         for t in graph.nodes:
#             if s != t:
#                 shortest_path_nodes = set()
#                 for sp in nx.all_shortest_paths(graph, s, t):
#                     for n in sp:
#                         shortest_path_nodes.add(n)
#                 bases.add(frozenset(shortest_path_nodes))
#     bases_list = list(bases)
#     node_to_bases = compute_node_to_bases(graph, bases_list, disable=disable)
#     return bases_list, node_to_bases

def get_log_balls_for_vertex(d_mat, node, tolerance=0.0):
    starting_node_ind = node
    ball_bases = set()
    # Sort the distance matrix based on the current node
    ind_to_sort = np.argsort(d_mat[starting_node_ind])
    d_node_mat = d_mat[:, ind_to_sort]
    # print(d_node_mat)
    dist = 0
    curr_set = set()
    temp = None
    sizes = [2**n - 1 for n in range(math.ceil(np.log2(len(d_mat))) + 1)]
    size_ind = 0
    
    for ind, el in enumerate(d_node_mat[starting_node_ind]):
        # add column element
        curr_node = ind_to_sort[ind]
        # if distance increasing more than tolerance change ball
        if el > dist + tolerance:
            dist = el
            temp = frozenset(curr_set)
        if ind > sizes[size_ind] and temp:
            ball_bases.add(temp)
            size_ind += 1
            temp = None
        curr_set.add(curr_node)
    # add last base
    ball_bases.add(frozenset(curr_set))
    return list(ball_bases)

def get_balls_for_vertex(d_mat, node, tolerance=0.0):
    starting_node_ind = node
    ball_bases = set()
    # Sort the distance matrix based on the current node
    ind_to_sort = np.argsort(d_mat[starting_node_ind])
    d_node_mat = d_mat[:, ind_to_sort]
    # print(d_node_mat)
    dist = 0
    curr_set = set()
    for ind, el in enumerate(d_node_mat[starting_node_ind]):
        # add column element
        curr_node = ind_to_sort[ind]
        # if distance increasing more than tolerance change ball
        if el > dist + tolerance:
            dist = el
            ball_bases.add(frozenset(curr_set))
        curr_set.add(curr_node)
    # add last base
    ball_bases.add(frozenset(curr_set))
    return list(ball_bases)

def retrieve_ball_bases(graph, distance, n2b = True, tolerance = 0.0, symmetric = False, disable = False, log_balls = False):
    d_mat = np.zeros((len(graph.nodes), len(graph.nodes)))
    for node_i in tqdm(graph.nodes, desc='Computing distance matrix', disable=disable):
        for node_j in graph.nodes:
            if symmetric and node_i == node_j:
                break
            d_mat[node_i][node_j] = distance(node_i, node_j)
    #print(d_mat)   
    if symmetric:
        for node_i in graph.nodes:
            for node_j in range(node_i, len(d_mat)):
                d_mat[node_i][node_j] = d_mat[node_j][node_i]
    
    full_bases = set()
    i = 0
    for node in tqdm(graph.nodes, desc="Retrieving bases", disable=disable):
        bases_n = get_log_balls_for_vertex(d_mat, node, tolerance) if log_balls else get_balls_for_vertex(d_mat, node, tolerance)
        for new_base in bases_n:
            full_bases.add(new_base)
        del bases_n
        
        # if i%100 == 0:

        #     # Get the sizes of all variables in the current scope
        #     sizes = {k: get_size(v) for k, v in locals().items()}
        #     # Sort the sizes dictionary by size in descending order
        #     sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
        #     # Print out the top 10 largest variables
        #     print("Top 10 largest variables:")
        #     for i, (name, size) in enumerate(sorted_sizes[:3]):
        #         print(f"{i+1}. {name}: {size*1e-9} GB")
        
        # i+=1
        
    bases_list = list(full_bases)
    if n2b:
        node_to_bases = compute_node_to_bases(graph, bases_list, disable=disable)
        return bases_list, node_to_bases
    
    return bases_list



def get_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_size(v) for v in obj.values())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_size(v) for v in obj)
    return size


def train_model(graph, bases_list, node_to_bases_ind, T, K_classes,  k_winnow = 1, epochs = 1, verbose = True, name='', seed = 42, disable_tqdm_train = False, disable_tqdm_epochs = False, debug = False):
    '''
    Train the multi-class-winnow algorithm on the graph, for T time steps, with given bases list
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    temp_pred = 0
    temp_upd = 0
    tot_mistakes = 0
    results = []
    # Create K winnow
    # winnows = [Winnow(bases_list) for _ in range(k)]
    # Stephen parameters
    m = T / 2
    winnows = [WinnowStephen(bases_list, K_classes, k_winnow, T, m) for _ in range(K_classes)]
    for _ in tqdm(range(epochs), desc="Epoch...", disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        # For each trial
        n_b = 0 
        for i in tqdm(range(T), desc=f"Training {name}", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict for each winnow
            temp = time.time()
            predictions = [(cls, win.predict(x_t, node_to_bases_ind)) for cls, win in enumerate(winnows)]
            temp_pred += time.time() - temp
            # Get true returns
            pred_classes = list(filter(lambda x: x[1]>=1/2, predictions))
            # Predict
            if pred_classes:
                # if len(pred_classes) > 1:
                #     print(f"MULTIPLE-CHOICES, Node: {x_t}")
                # y_hat_t = random_instance.choice(pred_classes)[0]
                n_b += 1
                y_hat_t = max(pred_classes, key=lambda x : x[1])[0]
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
            
            # Update winnow
            temp = time.time()
            for cls, pred in predictions:
                pred = int(pred >= 1/2)
                if cls == y_t:
                    if verbose:
                        print(f"update winnow for class: {cls}, predicted: {int(pred)}, true: {1}")
                    winnows[cls].update(x_t, pred, 1, node_to_bases_ind)
                else:
                    if verbose:
                        print(f"update winnow for class: {cls}, predicted: {int(pred)}, true: {0}")
                    winnows[cls].update(x_t, pred, 0, node_to_bases_ind)
            temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), winnows
    return tot_mistakes, results

def train_model_oldW(graph, bases_list, node_to_bases_ind, T, K_classes, epochs = 1, verbose = True, name='', seed = 42, disable_tqdm_train = False, disable_tqdm_epochs = False, debug = False):
    '''
    Train the multi-class-winnow algorithm on the graph, for T time steps, with given bases list
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    temp_pred = 0
    temp_upd = 0
    tot_mistakes = 0
    results = []
    # Create K winnow
    # winnows = [Winnow(bases_list) for _ in range(k)]
    # Stephen parameters
    m = T / 2
    winnows = [Winnow(bases_list) for _ in range(K_classes)]
    for _ in tqdm(range(epochs), desc="Epoch...", disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        # For each trial
        n_b = 0 
        for i in tqdm(range(T), desc=f"Training {name}", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict for each winnow
            temp = time.time()
            predictions = [(cls, win.predict(x_t, node_to_bases_ind)) for cls, win in enumerate(winnows)]
            temp_pred += time.time() - temp
            # Get true returns
            pred_classes = list(filter(lambda x: x[1]>=1/2, predictions))
            # Predict
            if pred_classes:
                # if len(pred_classes) > 1:
                #     print(f"MULTIPLE-CHOICES, Node: {x_t}")
                y_hat_t = random_instance.choice(pred_classes)[0]
                # n_b += 1
                # y_hat_t = max(pred_classes, key=lambda x : x[1])[0]
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
            
            # Update winnow
            temp = time.time()
            for cls, pred in predictions:
                # pred = int(pred >= 1/2)
                if cls == y_t:
                    if verbose:
                        print(f"update winnow for class: {cls}, predicted: {int(pred)}, true: {1}")
                    winnows[cls].update(x_t, pred, 1, node_to_bases_ind)
                else:
                    if verbose:
                        print(f"update winnow for class: {cls}, predicted: {int(pred)}, true: {0}")
                    winnows[cls].update(x_t, pred, 0, node_to_bases_ind)
            temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), winnows
    return tot_mistakes, results

def train_model_space_efficient(graph, bases_list, T, K_classes, epochs = 1, verbose = True, name='', seed = 42, disable_tqdm_train = False, disable_tqdm_epochs = False, debug = False):
    '''
    Train the multi-class-winnow algorithm on the graph, for T time steps, with given bases list
    '''
    random_instance = random.Random(seed)
    random_instance.seed(seed)
    temp_pred = 0
    temp_upd = 0
    tot_mistakes = 0
    results = []
    # Create K winnow
    # winnows = [Winnow(bases_list) for _ in range(k)]
    # Stephen parameters
    m = T / 2
    winnows = [WinnowSpaceEff(bases_list) for _ in range(K_classes)]
    for _ in tqdm(range(epochs), desc="Epoch...", disable=disable_tqdm_epochs):
        # Set to draw nodes from
        unlabeled_nodes = set(graph.nodes)
        # For each trial
        n_b = 0 
        for i in tqdm(range(T), desc=f"Training {name}", disable=disable_tqdm_train):
            if not unlabeled_nodes:
                break
            # Draw an unlabeled node and its label
            x_t = random_instance.choice(list(unlabeled_nodes))
            unlabeled_nodes.remove(x_t)
            y_t = graph.nodes[x_t]['label']
            
            # Predict for each winnow
            temp = time.time()
            predictions = [(cls, win.predict(x_t)) for cls, win in enumerate(winnows)]
            temp_pred += time.time() - temp
            # Get true returns
            pred_classes = list(filter(lambda x: x[1]>=1/2, predictions))
            # Predict
            if pred_classes:
                # if len(pred_classes) > 1:
                #     print(f"MULTIPLE-CHOICES, Node: {x_t}")
                # y_hat_t = random_instance.choice(pred_classes)[0]
                n_b += 1
                y_hat_t = max(pred_classes, key=lambda x : x[1])[0]
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
            
            # Update winnow
            temp = time.time()
            for cls, pred in predictions:
                pred = int(pred >= 1/2)
                if cls == y_t:
                    if verbose:
                        print(f"update winnow for class: {cls}, predicted: {int(pred)}, true: {1}")
                    winnows[cls].update(x_t, pred, 1)
                else:
                    if verbose:
                        print(f"update winnow for class: {cls}, predicted: {int(pred)}, true: {0}")
                    winnows[cls].update(x_t, pred, 0)
            temp_upd += time.time() - temp
    if verbose:
        print("Time for predictions: ", temp_pred)
        print("Time for updates: ", temp_upd)
        print(f"Training time for {name} : {temp_pred + temp_upd}")
    # print(f"TIES {name}: ", n_b)
    if debug:
        return (tot_mistakes, results), winnows
    return tot_mistakes, results