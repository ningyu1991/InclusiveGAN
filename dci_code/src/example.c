/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 * 
 * This code implements the method described in the Prioritized DCI paper, which 
 * can be found at https://arxiv.org/abs/1703.00440
 * 
 * Copyright (C) 2017    Ke Li
 * 
 * 
 * This file is part of the Dynamic Continuous Indexing reference implementation.
 * 
 * The Dynamic Continuous Indexing reference implementation is free software: 
 * you can redistribute it and/or modify it under the terms of the GNU Affero 
 * General Public License as published by the Free Software Foundation, either 
 * version 3 of the License, or (at your option) any later version.
 * 
 * The Dynamic Continuous Indexing reference implementation is distributed in 
 * the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 * See the GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with the Dynamic Continuous Indexing reference implementation.  If 
 * not, see <http://www.gnu.org/licenses/>.
 */

#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "dci.h"
#include "util.h"

int main(int argc, char **argv) {
    
    srand48(time(NULL));
    
    int j, k;
    
    int dim = 5000;
    int intrinsic_dim = 50;
    int num_points = 10000;
    int num_queries = 5;
    int num_neighbours = 10;    // The k in k-NN
    
    // Guide for tuning hyperparameters:
    
    // num_comp_indices trades off accuracy vs. construction and query time - high values lead to more accurate results, but slower construction and querying
    // num_simp_indices trades off accuracy vs. construction and query time - high values lead to more accurate results, but slower construction and querying; if num_simp_indices is increased, may need to increase num_comp_indices
    // num_levels trades off construction time vs. query time - higher values lead to faster querying, but slower construction; if num_levels is increased, may need to increase query_field_of_view and construction_field_of_view
    // construction_field_of_view trades off accuracy/query time vs. construction time - higher values lead to *slightly* more accurate results and/or *slightly* faster querying, but *slightly* slower construction
    // construction_prop_to_retrieve trades off acrruacy vs. construction time - higher values lead to *slightly* more accurate results, but slower construction
    // query_field_of_view trades off accuracy vs. query time - higher values lead to more accurate results, but *slightly* slower querying
    // query_prop_to_retrieve trades off accuracy vs. query time - higher values lead to more accurate results, but slower querying
    
    int num_comp_indices = 2;
    int num_simp_indices = 7;
    int num_levels = 2;
    int construction_field_of_view = 10;
    double construction_prop_to_retrieve = 0.002;
    int query_field_of_view = 100;
    double query_prop_to_retrieve = 0.05;
    
    
    // Generate data
    // Assuming column-major layout, data is dim x num_points
    double* data = (double *)memalign(64, sizeof(double)*dim*(num_points+num_queries));
    gen_data(data, dim, intrinsic_dim, num_points+num_queries);
    // Assuming column-major layout, query is dim x num_queries
    double* query = data + dim*((long long int)num_points);
    
    //print_matrix(data, dim, num_points);
    
    dci dci_inst;
        
    dci_init(&dci_inst, dim, num_comp_indices, num_simp_indices);
    
    //print_matrix(dci_inst.proj_vec, dim, num_comp_indices*num_simp_indices);
    
    dci_query_config construction_query_config;
    
    construction_query_config.blind = false;
    construction_query_config.num_to_visit = -1;
    construction_query_config.num_to_retrieve = -1;
    construction_query_config.prop_to_visit = 1.0;
    construction_query_config.prop_to_retrieve = construction_prop_to_retrieve;
    construction_query_config.field_of_view = construction_field_of_view;
    
    dci_add(&dci_inst, dim, num_points, data, num_levels, construction_query_config);
    
    // Query
    dci_query_config query_config;
    
    query_config.blind = false;
    query_config.num_to_visit = -1;
    query_config.num_to_retrieve = -1;
    query_config.prop_to_visit = 1.0;
    query_config.prop_to_retrieve = query_prop_to_retrieve;
    query_config.field_of_view = query_field_of_view;
    
    
    // Assuming column-major layout, matrix is of size num_neighbours x num_queries
    int** nearest_neighbours = (int **)malloc(sizeof(int *)*num_queries);
    double** nearest_neighbour_dists = (double **)malloc(sizeof(double *)*num_queries);
    int* num_returned = (int *)malloc(sizeof(int)*num_queries);
    
    dci_query(&dci_inst, dim, num_queries, query, num_neighbours, query_config, nearest_neighbours, nearest_neighbour_dists, num_returned);
    
    for (j = 0; j < num_queries; j++) {
        printf("%d: ", j+1);
        for (k = 0; k < num_returned[j]; k++) {
            printf("%d: %.4f, ", nearest_neighbours[j][k], nearest_neighbour_dists[j][k]);
        }
        printf("%d: %.4f\n", nearest_neighbours[j][num_neighbours-1], nearest_neighbour_dists[j][num_neighbours-1]);
    }
    
    for (j = 0; j < num_queries; j++) {
        free(nearest_neighbours[j]);
    }
    free(nearest_neighbours);
    for (j = 0; j < num_queries; j++) {
        free(nearest_neighbour_dists[j]);
    }
    free(nearest_neighbour_dists);
    free(num_returned);
    
    dci_free(&dci_inst);
    free(data);
    
    return 0;
}
