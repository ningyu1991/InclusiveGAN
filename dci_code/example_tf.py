import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())

from time import time

def construct_graph(dim, num_comp_indices = 2, num_simp_indices = 7, num_levels = 2, construction_prop_to_visit = 1.0, construction_prop_to_retrieve = 0.002, construction_field_of_view = 10, query_prop_to_visit = 1.0, query_prop_to_retrieve = 0.05, query_field_of_view = 100):
    dci_module = tf.load_op_library('./_dci_tf.so')
    graph = tf.Graph()
    with graph.as_default():
        query = tf.placeholder(tf.float64, shape = [None, None], name = "query")
        data = tf.placeholder(tf.float64, shape = [None, None], name = "data")
        num_neighbours = tf.placeholder(tf.int32, shape = [], name = "num_neighbours")
        update_db = tf.placeholder(tf.bool, shape = [], name = "update_db")
        nearest_neighbour_ids, nearest_neighbour_dists = dci_module.dci_nn_search(data, query, num_neighbours, update_db, dim = dim, num_comp_indices = num_comp_indices, num_simp_indices = num_simp_indices, num_levels = num_levels, construction_prop_to_visit = construction_prop_to_visit, construction_prop_to_retrieve = construction_prop_to_retrieve, construction_field_of_view = construction_field_of_view, query_prop_to_visit = query_prop_to_visit, query_prop_to_retrieve = query_prop_to_retrieve, query_field_of_view = query_field_of_view)
    placeholders = {"query": query, "data": data, "num_neighbours": num_neighbours, "update_db": update_db}
    outputs = {"nearest_neighbour_ids": nearest_neighbour_ids, "nearest_neighbour_dists": nearest_neighbour_dists}
    return graph, placeholders, outputs
    
def gen_data(ambient_dim, intrinsic_dim, num_points):
    latent_data = 2 * np.random.rand(num_points, intrinsic_dim) - 1     # Uniformly distributed on [-1,1)
    transformation = 2 * np.random.rand(intrinsic_dim, ambient_dim) - 1
    data = np.dot(latent_data, transformation)
    return data     # num_points x ambient_dim

def main(*args):
    
    dim = 5000
    intrinsic_dim = 50
    num_points = 10000
    num_queries = 5
    num_neighbours = 10    # The k in k-NN
    
    # Guide for tuning hyperparameters:
    
    # num_comp_indices trades off accuracy vs. construction and query time - high values lead to more accurate results, but slower construction and querying
    # num_simp_indices trades off accuracy vs. construction and query time - high values lead to more accurate results, but slower construction and querying; if num_simp_indices is increased, may need to increase num_comp_indices
    # num_levels trades off construction time vs. query time - higher values lead to faster querying, but slower construction; if num_levels is increased, may need to increase query_field_of_view and construction_field_of_view
    # construction_field_of_view trades off accuracy/query time vs. construction time - higher values lead to *slightly* more accurate results and/or *slightly* faster querying, but *slightly* slower construction
    # construction_prop_to_retrieve trades off acrruacy vs. construction time - higher values lead to *slightly* more accurate results, but slower construction
    # query_field_of_view trades off accuracy vs. query time - higher values lead to more accurate results, but *slightly* slower querying
    # query_prop_to_retrieve trades off accuracy vs. query time - higher values lead to more accurate results, but slower querying
    
    num_comp_indices = 2
    num_simp_indices = 7
    num_levels = 2
    construction_field_of_view = 10
    construction_prop_to_retrieve = 0.002
    query_field_of_view = 100
    query_prop_to_retrieve = 0.05
    
    
    print("Generating Data... ")
    t0 = time()
    data_and_queries = gen_data(dim, intrinsic_dim, num_points + num_queries)
    data = np.copy(data_and_queries[:num_points,:])
    queries = data_and_queries[num_points:,:]
    print("Took %.4fs" % (time() - t0))
    
    print("Constructing Graph... ")
    t0 = time()
    graph, placeholders, outputs = construct_graph(dim, num_comp_indices, num_simp_indices, num_levels = num_levels, construction_prop_to_retrieve = construction_prop_to_retrieve, construction_field_of_view = construction_field_of_view, query_prop_to_retrieve = query_prop_to_retrieve, query_field_of_view = query_field_of_view)
    print("Took %.4fs" % (time() - t0))

    print("Starting Tensorflow Session... ")
    t0 = time()
    with tf.Session(graph=graph) as sess:
        print("Took %.4fs" % (time() - t0))
        print("Constructing Data Structure and Querying Using Tensorflow... ")
        t0 = time()
        nearest_neighbour_ids, nearest_neighbour_dists = sess.run([outputs["nearest_neighbour_ids"], outputs["nearest_neighbour_dists"]], feed_dict={placeholders["data"]: data, placeholders["query"]: queries, placeholders["num_neighbours"]: num_neighbours, placeholders["update_db"]: True})
        print("Took %.4fs" % (time() - t0))
        
    print(nearest_neighbour_ids)
    #print(nearest_neighbour_dists)
    
if __name__ == '__main__':
    main(*sys.argv[1:])
