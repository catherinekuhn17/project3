# write tests for bfs
import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components



def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """ Helper function to check the correctness of the adjacency matrix encoding an MST.
        Note that because the MST of a graph is not guaranteed to be unique, we cannot 
        simply check for equality against a known MST of a graph. 

        Arguments:
            adj_mat: Adjacency matrix of full graph
            mst: Adjacency matrix of proposed minimum spanning tree
            expected_weight: weight of the minimum spanning tree of the full graph
            allowed_error: Allowed difference between proposed MST weight and `expected_weight`

        TODO: 
            Add additional assertions to ensure the correctness of your MST implementation
        For example, how many edges should a minimum spanning tree have? Are minimum spanning trees
        always connected? What else can you think of?
    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error
    
    num_unique_edges = len(np.unique(np.array([sorted((v1,v2)) 
                                                for v1, v2 in zip(np.where(mst!=0)[0], np.where(mst!=0)[1])]) , axis=0)) 
    
    assert num_unique_edges == len(adj_mat)-1 # number of edges in a MST should be # of nodes - 1.
    assert connected_components(mst)[0] == 1 # only 1 component in graph (meaning all connected)
    assert 0 not in [sum(row) for row in mst] # no rows/columns should be entirely empty

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """ Unit test for the construction of a minimum spanning tree on a small graph """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """ Unit test for the construction of a minimum spanning tree using 
    single cell data, taken from the Slingshot R package 
    (https://bioconductor.org/packages/release/bioc/html/slingshot.html)
    """
    file_path = './data/slingshot_example.txt'
    # load coordinates of single cells in low-dimensional subspace
    coords = np.loadtxt(file_path)
    # compute pairwise distances for all 140 cells to form an undirected weighted graph
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """ TODO: Write at least one unit test for MST construction """
    # Testing edge case where mst does not exist bc graph not fully connected
    file_path = './data/disconnected_matrix.csv'
    g = Graph(file_path)
    g.construct_mst()
    assert g.mst == None
    
    pass
