import warnings
import networkx as nx
from tqdm.auto import tqdm
import scipy

def dgl_to_nx_graph(dataset):
    # Get the graph and node labels
    graph = dataset[0]
    # Nodes and labels
    nodes = graph.nodes().tolist()
    labels = graph.ndata['label'].tolist()
    unique_labels = set(labels)
    # Edges
    src,dest = graph.edges()
    src = src.tolist()
    dest = dest.tolist()
    sd = zip(src,dest)
    # Create graph
    nx_graph = nx.DiGraph(sd)
    # Add the labels to the NetworkX graph
    node_labels = dict(zip(nodes, labels))
    nx.set_node_attributes(nx_graph, node_labels, 'label')
    return nx_graph, unique_labels

from collections import defaultdict

def retrieve_label_node_dict(Gnx, label_class_list):
    '''
    Helper to retrieve label and nodes for each label.
    ---
    Returns:
        label_class : [nodes in the label class]
    '''
    
    labels_node_list = {
        label_class : set([node_id for node_id, label in Gnx.nodes.data() if label['label'] == label_class] )
        for label_class in label_class_list
    }
    return labels_node_list

def node_class_betweeness_centrality(graph, unique_labels):
    '''
    Compute the conditional betweeness centrality of each node for each possible class.
    ---
    Returns:
        (node, true_label_class) : {label_class : bt_centrality_contitionate}

    '''
    class_cond_bt_cntr = {
        label_class : {
                node_id : 0.0 for node_id,_ in graph.nodes.data()
            } for label_class in unique_labels
    }
    
    labels_node_list = retrieve_label_node_dict(graph, unique_labels)
    
    for label_class, label_node_list in labels_node_list.items():
        label_btw_centr = nx.betweenness_centrality_subset(graph, label_node_list,label_node_list)
        for node_id, btw_cntr in label_btw_centr.items():
            assert class_cond_bt_cntr[label_class][node_id] == 0.0
            class_cond_bt_cntr[label_class][node_id] = btw_cntr
    # Structure output
    node_btw_cntr = {
        (node_id, true_label_class['label']) : {
            label_class : class_cond_bt_cntr[label_class][node_id] for label_class in unique_labels
        } for node_id, true_label_class in graph.nodes.data()
    }

    return node_btw_cntr

def evaluate_convexity(conv_dict):
    '''
    Evaluate the convexity of the dictionary in input.
    ---
    Input:
        (dict) (node, true_label_class) : {label_class : bt_centrality_contitionate}
    Return:
        - # Correctly evaluated / # Total nodes
        - (dict) label_class : (# Correctly evaluated nodes in that class / # Total nodes in the class)
    '''
    eval_dict = defaultdict(lambda: 0)
    eval_dict_tot = defaultdict(lambda: 0)
    correct = 0
    tot = 0
    for (node, true_class), bt_centr_dict in conv_dict.items():
        if (bt_centr_dict[true_class] == max(bt_centr_dict.values())):
            correct += 1
            eval_dict[true_class] += 1
        eval_dict_tot[true_class] += 1
        tot += 1
    assert tot == len(conv_dict), "error"
    tot_metric = correct / tot
    metric_dict = {
        label : val/eval_dict_tot[label] for label,val in eval_dict.items()
    }
    return tot_metric, metric_dict

def compute_spc(graph, unique_labels, verbose=True, disable=False):
    '''
    Compute the Shortest path convexity and the relaxed version for each label in the graph.
    ---
    Input:
        nx.Graph : graph with labels associated to each node as 'label' attribute
        unique_labels : list of the possible labels
    ---
    Returns:
        (dict) label_class :    # shortest paths in the class / # shortest_paths , 
                                # not reachable, 
                                # points in the shortest path within the class / # of points in the shortest path
    '''
    metric_dict = {label : [] for label in unique_labels}
    label_class_dict = retrieve_label_node_dict(graph, unique_labels)
    for label,node_set in tqdm(label_class_dict.items(), desc=f"Computing SPC for labels", disable=disable):
        # metrics for each label
        n_tot_path = 0
        n_conv_paths = 0
        not_reachable = 0
        tot_path_nodes = 0
        convex_path_nodes = 0
        node_list = list(node_set)
        for ind_n1 in tqdm(range(len(node_list)), desc="Iterating over nodes...", disable = disable):
            for ind_n2 in range(ind_n1 + 1, len(node_list)):
                node1,node2 = node_list[ind_n1],node_list[ind_n2]
                try:
                    for sp in nx.all_shortest_paths(graph, node1, node2):
                        convex_path = True
                        for n in sp:
                            if n in node_set:
                                convex_path_nodes += 1
                            tot_path_nodes +=1
                            convex_path = convex_path and (n in node_set)
                        n_tot_path += 1
                        if convex_path:
                            n_conv_paths += 1
                except:
                    not_reachable += 1
        if verbose:
            print("Label: ", label)
            print(n_tot_path)
            print(n_conv_paths)
            print(not_reachable)
        assert n_tot_path+not_reachable >= scipy.special.binom(len(node_list), 2), "Not correct evaluation"
        if tot_path_nodes == 0: 
            warnings.warn(f"Class {label} with a single node, counting it as convex")
            metric_dict[label] = (1.0,0.0,1.0)
        else:
            if n_tot_path > 0:
                metric_dict[label] = (n_conv_paths / n_tot_path , not_reachable, convex_path_nodes/tot_path_nodes)
            else:
                warnings.warn(f"No path is fully within class {label}")
                metric_dict[label] = (0,not_reachable, convex_path_nodes/tot_path_nodes)
    return metric_dict