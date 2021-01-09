'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in the Prioritized DCI paper, which 
can be found at https://arxiv.org/abs/1703.00440

Copyright (C) 2017    Ke Li


This file is part of the Dynamic Continuous Indexing reference implementation.

The Dynamic Continuous Indexing reference implementation is free software: 
you can redistribute it and/or modify it under the terms of the GNU Affero 
General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

The Dynamic Continuous Indexing reference implementation is distributed in 
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the Dynamic Continuous Indexing reference implementation.  If 
not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from dci import DCI

from time import time

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
    
    print("Constructing Data Structure... ")
    t0 = time()
    
    dci_db = DCI(dim, num_comp_indices, num_simp_indices)
    dci_db.add(data, num_levels = num_levels, field_of_view = construction_field_of_view, prop_to_retrieve = construction_prop_to_retrieve)
    
    print("Took %.4fs" % (time() - t0))
    
    print("Querying... ")
    t0 = time()
    
    nearest_neighbour_idx, nearest_neighbour_dists = dci_db.query(queries, num_neighbours = num_neighbours, field_of_view = query_field_of_view, prop_to_retrieve = query_prop_to_retrieve, blind = False)
    
    print("Took %.4fs" % (time() - t0))
    print(nearest_neighbour_idx)
    print(nearest_neighbour_dists)
    
if __name__ == '__main__':
    main(*sys.argv[1:])
