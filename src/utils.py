import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import jax.numpy as jnp
import networkx as nx
import random

def find_bases(bases_list, node_id):
    '''
    Return indexes of the bases elements that contains the node
    '''
    res_ind = []
    for ind, base in enumerate(bases_list):
        if node_id in base:
            res_ind.append(ind)
    assert len(res_ind) > 0, f"Invalid, no base for node {node_id}"
    return res_ind

def compute_edp_mat(graph, disable = False):
    '''
    Edge disjoint path distance (1/edp) matrix. Assumption on undirected graphs!
    '''
    gom_tree = nx.gomory_hu_tree(graph)
    gom_tree_path = dict(nx.shortest_path(gom_tree, weight="weight"))
    edp_mat = dict()
    for u in tqdm(graph.nodes, desc=f"Computing EDP matrix", disable=disable):
        edp_mat[u] = dict()
        for v in graph.nodes:
            if u == v:
                edp_mat[u][v] = 0
            else:
                edp_mat[u][v] = 1 / min((gom_tree[u][v]["weight"]) for (u, v) in zip(gom_tree_path[u][v], gom_tree_path[u][v][1:]))
    return edp_mat

def create_grid(n_rows, n_cols, capacity = 1.0):
    '''
    Create a 2d grid with a split in the labels row-wise.
    Returns Tuple:
      networkx.Graph: grid graph, dict: node-label dictionary (to draw)
    '''
    n_labels = 2

    grid = nx.grid_2d_graph(n_rows,n_cols)
    grid_l = dict()
    for r in range(n_rows):
        for i in range(n_cols):
            grid.nodes[r,i]['label'] = 0 if r < int(n_rows / 2) else 1
            grid_l[(r,i)] = 0 if r < int(n_rows / 2) else 1
    for edge in grid.edges:
        grid.edges[edge]['capacity'] = capacity
    return grid, grid_l

import math

def grid_graph(rows, cols):
    """
    Create a binary grid graph
    """
    # Initialize an empty list to store the edges
    edges = []
    # Loop over the rows and columns
    for i in range(rows):
        for j in range(cols):
            # Get the current node's index
            node_index = i * cols + j
            # Check if there is a node to the right
            if j < cols - 1:
                # Calculate the index of the right node
                right_index = node_index + 1
                # Add the edge between the current node and the right node
                edges.append((node_index, right_index))
            # Check if there is a node below
            if i < rows - 1:
                # Calculate the index of the node below
                down_index = node_index + cols
                # Add the edge between the current node and the node below
                edges.append((node_index, down_index))
    a = nx.Graph()
    a.add_edges_from(edges)
    for i in range(math.ceil(rows/2) * cols):
        a.nodes[i]['label'] = 0
    for i in range(math.ceil(rows/2) * cols, rows*cols):
        a.nodes[i]['label'] = 1
    for e in a.edges:
        a.edges[e]['capacity'] = 1.0
    return a

def line_graph(n, k, normal=0.00, capacity = 1.0):
    '''
    Create a line graph with n vertexes and k classes the normal class is not included in the k, final graph
    will have k+1 classes if the normal probability is given.
    '''
    # Create an empty graph object
    label_dict = dict()
    G = nx.Graph()
    
    nodes_for_label = int(n/k)
    
    for i in range(n):
        if np.random.rand()<normal:
            G.add_node(i, label = k)
            label_dict[i] = k
        else:
            G.add_node(i, label = int(i / nodes_for_label))
            label_dict[i] = int(i / nodes_for_label)

    # Add edges between nodes in a line way
    for i in range(n-1):
        G.add_edge(i, i+1)
    
    for edge in G.edges:
        G.edges[edge]['capacity'] = capacity

    return G, label_dict

def compute_pinv(graph, b = 0, c = 0):
    '''
    Compute pseudoinverse of the graph and the dictionary for accessing it from named nodes.
    Use b and c to get the kernel perceptron.
    Returns:
        np.array : pseudo_inverse of graph laplacian
        dict : nodes to index
        dict : index to nodes
    '''
    # Degree matrix
    degree_mat = np.zeros((len(graph.nodes),len(graph.nodes)))
    for i, (_,d) in enumerate(nx.degree(graph)):
        degree_mat[i][i] = d
    # JAX Pinv
    L = jnp.array(degree_mat - nx.to_numpy_array(graph))
    L_pinv = jnp.linalg.pinv(L)
    # If else to support kernel in perceptron
    if b == 0 and c==0:
        return L_pinv.tolist()
    res = L_pinv + b * jnp.ones(L_pinv.shape) + c * jnp.identity(L_pinv.shape[0])
    return res.tolist()

def compute_node_to_bases(graph, bases_list, disable = False):
    # Compute node to bases as initialization step
    node_to_bases_ind = dict()
    for node in tqdm(graph.nodes, desc=f"Computing node to base", disable=disable):
        node_to_bases_ind[node] = find_bases(bases_list, node)
    return node_to_bases_ind

from collections import deque

def cycle_erased_trajectory(graph, source, destinations):
    """
    Returns a cycle erased trajectory in the graph from the source to the set of destinations.
    If no cycle erased trajectory exists, returns None.
    """
    visited = set() # set to store visited nodes
    path = [] # list to store the current path

    # BFS search
    queue = deque([(source, path)])
    while queue:
        node, path = queue.popleft()

        # Check if the node has already been visited
        if node in visited:
            # Check if the node is part of the current path
            if node in path:
                # Remove the cycle by skipping this node
                continue
            else:
                # Node has been visited but is not part of the current path
                continue

        # Mark the node as visited and add it to the current path
        visited.add(node)
        path.append(node)

        # Check if the node is a destination
        if node in destinations:
            # Backtrack to find the entire path and return it
            return path

        # Add unvisited neighbors to the queue
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, path[:]))

    # No cycle erased trajectory found
    return None

def wilson_random_spanning_tree(graph, seed = 42):
    random.seed(seed)
    visited = set()
    not_visited = set(list(graph.nodes))
    tree = nx.Graph()

    # Take root
    curr_node = not_visited.pop()
    visited.add(curr_node)

    # print("start from: ", curr_node)

    i = 0

    while not_visited:
        # Random chose a node in not visited 
        node_from = random.choice(list(not_visited))

        #print("NODE FROM: ", node_from)
        not_visited.remove(node_from)

        lerw = cycle_erased_trajectory(graph, node_from, visited)
        
        visited.add(node_from)
        #print("LEWR: ", lerw)
        not_visited.add(lerw[-1])
        for n in lerw[1:]:
            #print("VISITING ", n, " FROM ", node_from)
            not_visited.remove(n)
            visited.add(n)
            tree.add_edge(node_from, n)
            node_from = n
    return tree
    

def find_best_indexes(winnow):
    res_ind = []
    m = max(winnow.w)
    for i, val in enumerate(winnow.w):
        if val == m:
            res_ind.append(i)
    return res_ind

def find_best_bases(winnow):
    best_i = find_best_indexes(winnow)
    res = [winnow.bases_list[i] for i in best_i]
    return res

def retrieve_best_base_color(winnow, n_nodes):
    best_0 = find_best_bases(winnow)
    base_0_colors = [1 for _ in range(n_nodes)]
    print(f"Number of best bases: ", len(best_0))
    for bb in best_0:
        for ind in bb:
            base_0_colors[int(ind)] = 0
    return base_0_colors

def retrivene_colors_for_elements_base(winnow, node, n_nodes):
    best_0 = find_best_bases(winnow)
    base_0_colors = [1 for _ in range(n_nodes)]
    print(f"Number of best bases: ", len(best_0))
    for bb in best_0:
        for ind in bb:
            base_0_colors[int(ind)] = 0
    return base_0_colors

def inject_noise(graph, n_noise, seed = 42):
    res = graph.copy()
    random.seed(seed)
    dev = 1 / np.sqrt(len(res.nodes))
    for noise_p in range(len(res.nodes), len(res.nodes) + n_noise):
        res.add_node(noise_p)
        res.nodes[noise_p]['label'] = -1
        for n in range(len(res.nodes)):
            if random.random() <= dev:
                res.add_edge(n, noise_p)
    return res

def relabel_graph(graph):
    graph = graph.copy()
    labels = set([graph.nodes[n]['label'] for n in graph.nodes])
    reass_dict = dict()
    if -1 in labels:
        reass_dict[-1] = -1
        labels.remove(-1)
    
    for ind,l in enumerate(labels):
        reass_dict[l] = ind
    for n in graph.nodes:
        graph.nodes[n]['label'] = reass_dict[graph.nodes[n]['label']]
    return graph

def create_noise_graph(graph, labels_to_noise):
    G = graph.copy()
    for n in G.nodes:
        if G.nodes[n]['label'] in labels_to_noise:
            G.nodes[n]['label'] = -1
    return G

def create_noise_graph_perc(graph, labels_to_noise, perc, seed = 42):
    random.seed(seed)
    G = graph.copy()
    for n in G.nodes:
        if G.nodes[n]['label'] in labels_to_noise:
            G.nodes[n]['label'] = -1
        else:
            if random.random()<= perc:
                G.nodes[n]['label'] = -1
    return G

def relabel_nodes_id(graph):
    return nx.relabel_nodes(graph, {n: i for i, n in enumerate(graph.nodes())})

def create_convex_graph(rows, cols):
    # Initialize an empty list to store the edges
    edges = []
    # Loop over the rows and columns
    for i in range(rows):
        for j in range(cols):
            # Get the current node's index
            node_index = i * cols + j
            # Check if there is a node to the right
            if j < cols - 1:
                # Calculate the index of the right node
                right_index = node_index + 1
                # Add the edge between the current node and the right node
                edges.append((node_index, right_index))
            # Check if there is a node below
            if i < rows - 1:
                # Calculate the index of the node below
                down_index = node_index + cols
                # Add the edge between the current node and the node below
                edges.append((node_index, down_index))
    a = nx.Graph()
    a.add_edges_from(edges)
    for i in range(rows):
        for j in range(cols):
            if i >= 0.2*rows and i < 0.8*rows and j >= 0.2*cols and j < 0.8*cols:
                a.nodes[i*cols+j]['label'] = 0
            else:
                a.nodes[i*cols+j]['label'] = -1
    for e in a.edges:
        a.edges[e]['capacity'] = 1.0
    return a

def render_graph(testg, node_size = 1000, pos = None, path = None):
    fig, ax = plt.subplots(figsize= (24,12))
    node_colors = [testg.nodes[node]['label'] for node in testg.nodes]
    if not pos:
        pos = nx.spring_layout(testg, seed = 42)
    cmap = plt.cm.Accent
    nx.draw(
        testg, pos=pos, node_size=node_size, node_color = node_colors, 
        cmap = cmap, ax = ax, edge_color = '#131516', width = 0.005, edgecolors='black'
    )

    # sm = plt.cm.ScalarMappable(cmap=cmap)
    # cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    # sm.set_array(node_colors)
    # cbar = plt.colorbar(sm, cax=cax)
    # cbar.set_label('Node Label')

    if path is not None:
        plt.savefig(path)

    plt.show()

def create_random_graph(m, seed = 42, prob = 0.5, randA = False, randB = False, deterministic = False):
    """
    m : number of elements for each class
    """
    assert randA or randB or deterministic, "At least one type"
    random.seed(42)
    div = 1 / np.sqrt(m)
    G = nx.Graph()
    c1 = list(range(m))
    c2 = list(range(m, 2*m))
    # Class
    for i in c1:
        G.add_node(i)
        G.nodes[i]['label'] = 0
    for i in range(m):
        for j in range(i,m):
            if i != j:
                if random.random() <= prob:
                    G.add_edge(i,j)
    
    for i in c2:
        G.add_node(i)
        G.nodes[i]['label'] = -1
        # Random A edges
        if randA:
            for j in c1:
                if random.random() <= div:
                    G.add_edge(i,j)
        if randB:
            # Random B edges
            for j in range(2*m):
                if random.random() <= div and i != j:
                    G.add_edge(i,j)
    
    # DETERMINISTIC
    if deterministic:
        print("Deterministic")
        for i,j in zip(c1,c2):
            G.add_edge(i,j)
    
    
    nx.set_edge_attributes(G, 1.0, name = 'capacity')
    return G

from sklearn.metrics import pairwise_distances

def create_synth_graph_from_sample(dataset_test, labels, k, metric='euclidean', capacity = 1.0):
    '''
    Return knn graph
    '''
    X_flat = np.array(dataset_test)
    # Compute the similarity between all pairs of points
    distances = pairwise_distances(X_flat, metric=metric)
    # Get the indices of the nearest neighbor for each point
    nearest_indices = np.argsort(distances, axis=1)[:,1:k+1]
    G = nx.Graph()
    for n in range(X_flat.shape[0]):
        G.add_node(n)
        G.nodes[n]['label'] = int(labels[n])
        for nn in nearest_indices[n]:
            G.add_edge(n,nn)
    
    nx.set_edge_attributes(G, 1.0, 'capacity')
    return G

def create_epsilon_graph_from_sample(dataset_test, labels, epsilon, metric='euclidean', capacity = 1.0):
    '''
    Return epsilon graph
    '''
    # Reshape the array (n_samples, 28*28)
    X_flat = np.array(dataset_test)
    # Compute the similarity between all pairs of points
    distances = pairwise_distances(X_flat, metric=metric)
    G = nx.Graph()
    for n in range(X_flat.shape[0]):
        G.add_node(n)
        G.nodes[n]['label'] = int(labels[n])
        
    # Espilon balls:
    for ns in range(X_flat.shape[0]):
        for nd in range(ns + 1, X_flat.shape[0]):
            if distances[ns][nd] < epsilon:
                G.add_edge(ns,nd)
    
    while not nx.is_connected(G):
        cc_list = list(nx.connected_components(G))
        c1 = np.array([a for a in cc_list[0]])
        c2 = np.array([a for a in cc_list[1]])
        res = distances[c1][:,c2]
        min_idx = np.argmin(res)
        min_row, min_col = np.unravel_index(min_idx, res.shape)
        el1 = c1[min_row]
        el2 = c2[min_col]
        G.add_edge(el1,el2)
    

    nx.set_edge_attributes(G, 1.0, 'capacity')
    return G

def create_gaussian_graph(n_corner, n_center, sigma_corner, sigma_center, n_neighbors=0, epsilon = 0, seed = 42):
    """
    Return the square gaussian graph (not necessarly connected) in both NN and Epsilon formats.
    """
    np.random.seed(seed)
    mu_corner1 = [1, 1]
    mu_corner2 = [-1, 1]
    mu_corner3 = [1, -1]
    mu_corner4 = [-1, -1]
    mu_center = [0, 0]

    points_corner1 = np.random.normal(mu_corner1, sigma_corner, size=(n_corner, 2))
    points_corner2 = np.random.normal(mu_corner2, sigma_corner, size=(n_corner, 2))
    points_corner3 = np.random.normal(mu_corner3, sigma_corner, size=(n_corner, 2))
    points_corner4 = np.random.normal(mu_corner4, sigma_corner, size=(n_corner, 2))
    points_center = np.random.normal(mu_center, sigma_center, size=(n_center, 2))
    tot = np.concatenate(
    (points_corner1, points_corner2, points_corner3, points_corner4, points_center)
    )
    
    labels = [0 for _ in range(n_corner)] + [1 for _ in range(n_corner)] + [2 for _ in range(n_corner)] + [3 for _ in range(n_corner)] + [-1 for _ in range(n_center)]
    graph_nn = create_synth_graph_from_sample(tot, labels, n_neighbors)
    graph_eps = create_epsilon_graph_from_sample(tot, labels, epsilon)
    
    

    return graph_nn, graph_eps

def create_multi_class_clique(n_clique, classes, n_nodes_noise, rand_inst, shading = 0.0):
    clique = []
    G = nx.Graph()
    for i in range(classes):
        clique += [np.arange(n_clique) + i*n_clique]
    for i,c in enumerate(clique):
        G.add_nodes_from(c, label = i)
    for c in clique:
        G.add_edges_from([(node1, node2) for node1 in c for node2 in c if node1 != node2])
    
    p_noise = 1/np.sqrt(n_clique * classes) * shading
    
    nodes_id = n_clique * classes
    for _ in range(n_clique * classes, n_clique * classes + n_nodes_noise):
        G.add_node(nodes_id, label=-1)
        for n_c in G.nodes:
            if rand_inst.random() <= p_noise and n_c != nodes_id:
                G.add_edge(nodes_id, n_c) 
        if G.degree(nodes_id) > 0:
            nodes_id += 1
    nx.set_edge_attributes(G, 1.0, 'capacity')
    return G

def render_plot(loaded_data, num_rows, num_cols, figx = 16, figy = 16, save_fig = False, path = None):
    baselines = set(['CBSim', 'GABA', 'EXP3'])
    input_list = loaded_data[0]
    plt_np_arr = loaded_data[1]
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figx, figy))

    for i, (key, sub_dict) in enumerate(plt_np_arr.items()):
        row = i // num_cols
        col = i % num_cols

        for sub_key, sub_data in sub_dict.items():
            if sub_key in baselines:
                line =  'dashdot'
            else:
                line = 'solid'
            axs[row, col].plot(sub_data, label=sub_key, linestyle = line, lw = 2)
        
        axs[row, col].set_title(f"Cliques of {input_list[i][0]} nodes and {input_list[i][1]} nodes as noise")
        # axs[row, col].set_title(f"Cora, labels {input_list[i]} as noise",fontsize=15)
        # axs[row, col].set_title(f"Foreground {input_list[i][0]},{input_list[i][2]}; Background {input_list[i][1]},{input_list[i][3]};{input_list[i][4]}-NN")
        axs[row, col].set_xlabel('Time',fontsize=15)
        axs[row, col].set_ylabel('Mistakes',fontsize=15) 
        axs[row, col].grid()
    axs[0, 0].legend(fontsize=15)
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        fig.delaxes(axs[row, col])
    if save_fig:
        fig.savefig(path, format='svg')

def render_plot_CI(type, loaded_data, num_rows, num_cols, figx = 16, figy = 16, save_fig = False, path = None):
    input_list = loaded_data[0]
    plt_np_arr = loaded_data[1]
    
    baselines = set(['CBSim', 'GABA', 'EXP3'])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figx, figy))

    for i, (key, sub_dict) in enumerate(plt_np_arr.items()):
        row = i // num_cols
        col = i % num_cols
        for sub_key, sub_data in sub_dict.items():
            if sub_key in baselines:
                line =  'dashdot'
            else:
                line = 'solid'
            plot_data = np.mean(sub_data, axis = 0)
            axs[row, col].plot(plot_data, label=sub_key, linestyle = line, lw = 2)
            
            conf_int = np.std(sub_data, axis = 0) / np.sqrt(len(sub_data)) * 1.96
            axs[row, col].fill_between(np.arange(len(plot_data)), plot_data - conf_int, plot_data + conf_int, alpha=0.2)

        if type == 'Clique':
            axs[row, col].set_title(f"Cliques of {input_list[i][0]} nodes and {input_list[i][1]} nodes as noise", fontsize = 14)
        if type == 'Cora':
            axs[row, col].set_title(f"Cora, labels {input_list[i]} as noise",fontsize=15)
        if type == 'Citeseer':
            axs[row, col].set_title(f"Citeseer, labels {input_list[i]} as noise",fontsize=15)
        if type == 'Gaussian':
            axs[row, col].set_title(f"Foreground {input_list[i][0]},{input_list[i][2]}; Background {input_list[i][1]},{input_list[i][3]};{input_list[i][4]}-NN", fontsize = 14)
        if type == 'LastFM':
            axs[row, col].set_title(f"LastFM, labels {input_list[i]} as noise",fontsize=15)
        axs[row, col].set_xlabel('Time',fontsize=15)
        axs[row, col].set_ylabel('Mistakes',fontsize=15) 
        axs[row, col].grid()
    axs[0, 0].legend(fontsize=15)
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        fig.delaxes(axs[row, col])
    if save_fig:
        fig.savefig(path, format='svg')

def save_plots(loaded_data, path):
    baselines = set(['CBSim', 'GABA', 'EXP3'])
    
    input_list = loaded_data[0]
    plt_np_arr = loaded_data[1]
    for i in plt_np_arr.keys():
        for lab, val in plt_np_arr[i].items():
            if lab in baselines:
                line =  'dashdot'
            else:
                line = 'solid'
            plt.plot(val, label=lab,linestyle = line)
        plt.legend()
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Mistakes') 
        plt.savefig(f'{path}/{i}.svg', format='svg')
        plt.show()

def get_average_plot_cora(loaded_data, path):
    baselines = set(['CBSim', 'GABA', 'EXP3'])
    plt_np_arr = loaded_data[1]
    res = {
        lab : np.zeros(len(item)) for lab, item in plt_np_arr[frozenset([0, 1, 4])].items()
    }
    for _, d in plt_np_arr.items():
        for label, value in d.items():
            res[label] += value

    for key, val in res.items():
        res[key] = val/len(plt_np_arr)
    for lab, val in res.items():
        if lab in baselines:
                line =  'dashdot'
        else:
            line = 'solid'
        plt.plot(val, label=lab,linestyle = line)
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Mistakes')
    plt.savefig(path)
    plt.show()