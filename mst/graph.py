import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        Method to find the minimum spanning tree of a given adjacency matrix
        
        returns:
        -------
        an diagonally symmetric adjacency matrix of a minimum spanning tree
        
        """
        self.adj_mat = np.where(self.adj_mat == 0 , float('inf'), self.adj_mat) # setting edges wighted 0 to inf (since they don't exist so we can't use them)
        pq = [] # priority queue of vertices
        mst = np.zeros((len(self.adj_mat), len(self.adj_mat))) # initial output mst matrix of all zeroes
        v1 = np.random.choice(range(len(self.adj_mat))) # choosing a random starting vertex
       
        # creating a list of tuples of format (weight, vertex1, vertex2), to keep track of edge weights and the vertices connected
        tup_list = list(zip(self.adj_mat[v1], range(len(self.adj_mat)), np.ones(len(self.adj_mat), dtype=np.int64)*v1)) 
        visited_vertices = [v1] 
        for value in tup_list:
            heapq.heappush(pq, value) # adding v1 nbrs to priority queue

        while len(visited_vertices) < len(self.adj_mat): # loop through until we've visited all vertices
            lowest_weight = heapq.heappop(pq) # pop lowest weight edge
            edge_weight = lowest_weight[0] 
            v_out = lowest_weight[1]
            v_in = lowest_weight[2]

            if v_out not in visited_vertices: # if vertex we're going to not in visited vertices,
                visited_vertices.append(v_out) # add this vertex
                mst[v_in][v_out] = edge_weight # add lowest weight edge to mst
                mst[v_out][v_in] = edge_weight
                # Now create a new list of tuples of the weight of edges leaving this vertex, and push to the priority queue
                tup_list = list(zip(self.adj_mat[lowest_weight[1]], range(len(self.adj_mat)), np.ones(len(self.adj_mat), dtype=np.int64)*lowest_weight[1]))  
                for value in tup_list:
                    heapq.heappush(pq, value) # add new edge weights to tuple
        
        # Edge Case: input matrix is not fully connected (as in has more than one connected component). In this case,
        # since I reweighted the nonexistant edges to be infinity, infinity will be added to the mst matrix. Therefore, if there
        # is infinity in the final mst, an mst doesn't exist, as it can't be created with real edges.
        
        if float('inf') in mst:
            print('Input matrix is not connected. No minimum spanning tree could be found.')
            return None
        
        self.mst = mst