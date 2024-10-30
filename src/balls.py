
import networkx as nx
import numpy as np
import jax.numpy as jnp
from src.utils import compute_pinv, compute_edp_mat

class BallDistances:
    def __init__(self, graph, initialize_pinv = False, initialize_sp = False, initialize_edp = False, disable = False):
        self.graph = graph
        self.initialize_pinv = initialize_pinv
        self.initialize_sp = initialize_sp
        self.initialize_edp = initialize_edp
        self.disable = disable
        if self.initialize_pinv:
            print("Initializing pseudo inv...")
            self.L_pinv = compute_pinv(self.graph)
        if self.initialize_sp:
            print("Initializing shortest paths...")
            self.sp_mat = dict(nx.shortest_path_length(self.graph))
        if self.initialize_edp:
            print("Initializing edge disjoint paths...")
            self.edp_mat = compute_edp_mat(self.graph, disable = self.disable)

    def d1_ball_distance(self, u, v):
        if not self.initialize_edp:
            self.initialize_edp = True
            self.edp_mat = compute_edp_mat(self.graph, disable=self.disable) 
        return self.edp_mat[u][v]

    def d2_ball_distance(self, source, target):
        # Compute pinv if not already computed
        if not self.initialize_pinv:
            self.initialize_pinv = True
            self.L_pinv = compute_pinv(self.graph) 
            
        if source == target:
            return 0.0
        
        eff_res =   self.L_pinv[source][source] + \
                    self.L_pinv[target][target] - \
                    2 * self.L_pinv[source][target]
        
        return eff_res

    def dinf_ball_distance(self, source, target):
        if not self.initialize_sp:
            self.initialize_sp = True
            self.sp_mat = dict(nx.shortest_path_length(self.graph))
        if source == target:
            return 0.0
        return self.sp_mat[source][target]