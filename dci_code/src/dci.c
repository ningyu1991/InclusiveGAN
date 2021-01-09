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
#include <math.h>
#include <assert.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include "dci.h"
#include "util.h"

static inline double abs_d(double x) {
    return x > 0 ? x : -x;
}

static inline int min_i(int a, int b) {
    return a < b ? a : b;
}

static inline int max_i(int a, int b) {
    return a > b ? a : b;
}

typedef struct tree_node {
    int parent;
    int child;
} tree_node;

static void dci_gen_proj_vec(double* const proj_vec, const int dim, const int num_indices) {
    int i, j;
    double sq_norm, norm;
    for (i = 0; i < dim*num_indices; i++) {
        proj_vec[i] = rand_normal();
    }
    for (j = 0; j < num_indices; j++) {
        sq_norm = 0.0;
        for (i = 0; i < dim; i++) {
            sq_norm += (proj_vec[i+j*dim] * proj_vec[i+j*dim]);
        }
        norm = sqrt(sq_norm);
        for (i = 0; i < dim; i++) {
            proj_vec[i+j*dim] /= norm;
        }
    }
}

void dci_init(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices) {
    
    int num_indices = num_comp_indices*num_simp_indices;
    
    srand48(time(NULL));
    
    dci_inst->dim = dim;
    dci_inst->num_comp_indices = num_comp_indices;
    dci_inst->num_simp_indices = num_simp_indices;
    dci_inst->num_points = 0;
    dci_inst->num_levels = 0;
    dci_inst->num_coarse_points = 0;
    
    dci_inst->proj_vec = (double *)memalign(64, sizeof(double)*dim*num_indices);
    dci_inst->indices = NULL;
    dci_inst->data = NULL;
    dci_inst->next_level_ranges = NULL;
    dci_inst->num_finest_level_points = NULL;
    dci_gen_proj_vec(dci_inst->proj_vec, dim, num_indices);
    
}

static int dci_compare_idx_elem(const void *a, const void *b) {
    double key_diff = ((idx_elem *)a)->key - ((idx_elem *)b)->key;
    return (key_diff > 0) - (key_diff < 0);
}

static int dci_compare_tree_node(const void *a, const void *b) {
    return ((tree_node *)a)->parent - ((tree_node *)b)->parent;
}

static void dci_assign_parent(dci* const dci_inst, const int num_populated_levels, const int num_queries, const int *selected_query_pos, const double* const query, const double* const query_proj, const dci_query_config query_config, tree_node* const assigned_parent);

// Note: the data itself is not kept in the index and must be kept in-place
// Added data must be contiguous
void dci_add(dci* const dci_inst, const int dim, const int num_points, const double* const data, const int num_levels, const dci_query_config construction_query_config) {
    int h, i, j;
    int actual_num_levels, num_points_on_upper_levels, num_points_on_upper_and_cur_levels;
    // Only populated when actual_num_levels >= 2
    int **level_members;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    double *data_proj = (double *)memalign(64, sizeof(double)*num_indices*num_points);  // (# of indices) x (# of points) column-major when actual_num_levels >= 2, (# of points) x (# of indices) otherwise
    bool data_proj_transposed = false;  // True if data_proj is (# of points) x (# of indices) column-major; used only for error-checking
    tree_node *assigned_parent;
    int *data_levels;
    double promotion_prob;
    int num_points_on_level[num_levels];
    int level_relabelling[num_levels];
    
    assert(dim == dci_inst->dim);
    assert(dci_inst->num_points == 0);
    
    dci_inst->data = data;
    dci_inst->num_points = num_points;
    
    if (num_levels < 2) {
        
        num_points_on_level[0] = num_points;
        actual_num_levels = num_levels;
        
        level_members = NULL;
        
    } else {
        
        data_levels = (int *)malloc(sizeof(int)*num_points);
        promotion_prob = pow((double)num_points, -1.0 / num_levels);
        
        for (i = 0; i < num_levels; i++) {
            num_points_on_level[i] = 0;
        }
        for (j = 0; j < num_points; j++) {
            for (i = 0; i < num_levels - 1; i++) {
                if (drand48() > promotion_prob) {
                    break;
                }
            }
            num_points_on_level[i]++;
            data_levels[j] = i;
        }
        
        // Remove all levels with no points
        h = 0;
        for (i = 0; i < num_levels; i++) {
            if (num_points_on_level[i] > 0) {
                level_relabelling[i] = h;
                h++;
            } else {
                level_relabelling[i] = -1;
            }
        }
        actual_num_levels = h;
    
        for (i = 0; i < num_levels; i++) {
            if (level_relabelling[i] >= 0) {
                num_points_on_level[level_relabelling[i]] = num_points_on_level[i];
            }
        }
        
        if (actual_num_levels >= 2) {
        
            level_members = (int **)malloc(sizeof(int*)*actual_num_levels);
            for (i = 0; i < actual_num_levels; i++) {
                level_members[i] = (int *)malloc(sizeof(int)*num_points_on_level[i]);
                h = 0;
                for (j = 0; j < num_points; j++) {
                    if (level_relabelling[data_levels[j]] == i) {   
                        level_members[i][h] = j;
                        h++;
                    }
                }
                assert(h == num_points_on_level[i]);
            }
        
        } else {
            level_members = NULL;
        }
        
        free(data_levels);
        
    }

    dci_inst->num_coarse_points = num_points_on_level[actual_num_levels - 1];    
    dci_inst->num_levels = actual_num_levels;
    
    dci_inst->indices = (idx_elem **)malloc(sizeof(idx_elem*)*actual_num_levels);
    num_points_on_upper_and_cur_levels = 0;
    for (i = actual_num_levels - 1; i >= 0; i--) {
        num_points_on_upper_and_cur_levels += num_points_on_level[i];
        dci_inst->indices[i] = (idx_elem *)malloc(sizeof(idx_elem)*num_points_on_upper_and_cur_levels*num_indices);
    }

    dci_inst->next_level_ranges = (range **)malloc(sizeof(range*)*actual_num_levels);
    num_points_on_upper_and_cur_levels = 0;
    for (i = actual_num_levels - 1; i >= 1; i--) {
        num_points_on_upper_and_cur_levels += num_points_on_level[i];
        dci_inst->next_level_ranges[i] = (range *)malloc(sizeof(range)*num_points_on_upper_and_cur_levels);
    }
    dci_inst->next_level_ranges[0] = NULL;
    
    i = actual_num_levels - 1;
    num_points_on_upper_and_cur_levels = num_points_on_level[i];
    
    if (actual_num_levels < 2) {
        
        assigned_parent = NULL;
        
        // data_proj is (# of points) x (# of indices) column-major
        matmul(num_points, num_indices, dci_inst->dim, data, dci_inst->proj_vec, data_proj);
        data_proj_transposed = true;
        
        for (j = 0; j < num_indices*num_points_on_upper_and_cur_levels; j++) {
            dci_inst->indices[i][j].key = data_proj[j];
            dci_inst->indices[i][j].local_value = j % num_points_on_upper_and_cur_levels;
            dci_inst->indices[i][j].global_value = j % num_points_on_upper_and_cur_levels;
        }
        
    } else {
        
        assigned_parent = (tree_node *)malloc(sizeof(tree_node)*num_points);
        
        // data_proj is (# of indices) x (# of points) column-major
        matmul(num_indices, num_points, dci_inst->dim, dci_inst->proj_vec, data, data_proj);
        for (j = 0; j < num_points_on_upper_and_cur_levels; j++) {
            assigned_parent[j].child = level_members[i][j];
        }
        
        for (j = 0; j < num_points_on_upper_and_cur_levels; j++) {
            int k;
            for (k = 0; k < num_indices; k++) {
                dci_inst->indices[i][j+k*num_points_on_upper_and_cur_levels].key = data_proj[k+level_members[i][j]*num_indices];
                dci_inst->indices[i][j+k*num_points_on_upper_and_cur_levels].local_value = j;
                dci_inst->indices[i][j+k*num_points_on_upper_and_cur_levels].global_value = level_members[i][j];
            }
        }
    }
    
    #pragma omp parallel for
    for (j = 0; j < num_indices; j++) {
        qsort(&(dci_inst->indices[i][j*num_points_on_level[i]]), num_points_on_level[i], sizeof(idx_elem), dci_compare_idx_elem);
    }
    
    num_points_on_upper_levels = num_points_on_upper_and_cur_levels;
    
    for (i = actual_num_levels - 2; i >= 0; i--) {
        
        assert(!data_proj_transposed);
        
        for (j = 0; j < num_points_on_upper_levels; j++) {
            assigned_parent[j].parent = j;
        }
        
        dci_assign_parent(dci_inst, actual_num_levels - i - 1, num_points_on_level[i], level_members[i], data, data_proj, construction_query_config, &(assigned_parent[num_points_on_upper_levels]));
        
        num_points_on_upper_and_cur_levels = num_points_on_upper_levels + num_points_on_level[i];
        
        qsort(assigned_parent, num_points_on_upper_and_cur_levels, sizeof(tree_node), dci_compare_tree_node);
        
        h = 0;
        dci_inst->next_level_ranges[i+1][0].start = 0;
        for (j = 0; j < num_points_on_upper_and_cur_levels; j++) {
            if (assigned_parent[j].parent > h) {
                dci_inst->next_level_ranges[i+1][h].num = j - dci_inst->next_level_ranges[i+1][h].start;
                assert(dci_inst->next_level_ranges[i+1][h].num > 0);
                h++;
                assert(assigned_parent[j].parent == h);
                dci_inst->next_level_ranges[i+1][h].start = j;
            }
        }
        dci_inst->next_level_ranges[i+1][h].num = num_points_on_upper_and_cur_levels - dci_inst->next_level_ranges[i+1][h].start;
        assert(h == num_points_on_upper_levels - 1);
        
        for (j = 0; j < num_points_on_upper_and_cur_levels; j++) {
            range cur_indices_range = dci_inst->next_level_ranges[i+1][assigned_parent[j].parent];
            int k;
            for (k = 0; k < num_indices; k++) {
                dci_inst->indices[i][(j-cur_indices_range.start)+k*cur_indices_range.num+cur_indices_range.start*num_indices].key = data_proj[k+assigned_parent[j].child*num_indices];
                dci_inst->indices[i][(j-cur_indices_range.start)+k*cur_indices_range.num+cur_indices_range.start*num_indices].local_value = j - cur_indices_range.start;
                dci_inst->indices[i][(j-cur_indices_range.start)+k*cur_indices_range.num+cur_indices_range.start*num_indices].global_value = assigned_parent[j].child;
            }
        }
        
        #pragma omp parallel for
        for (j = 0; j < num_points_on_upper_levels*num_indices; j++) {
            range cur_indices_range = dci_inst->next_level_ranges[i+1][j / num_indices];
            int k = j % num_indices;
            qsort(&(dci_inst->indices[i][k*cur_indices_range.num+cur_indices_range.start*num_indices]), cur_indices_range.num, sizeof(idx_elem), dci_compare_idx_elem);
        }
        
        num_points_on_upper_levels = num_points_on_upper_and_cur_levels;
        
    }
    assert(num_points_on_upper_levels == num_points);
    
    // Populate dci_inst->num_finest_level_points
    dci_inst->num_finest_level_points = (int **)malloc(sizeof(int*)*actual_num_levels);
    dci_inst->num_finest_level_points[0] = NULL;
    if (actual_num_levels >= 2) {
        num_points_on_upper_and_cur_levels = num_points - num_points_on_level[0];
        dci_inst->num_finest_level_points[1] = (int *)malloc(sizeof(int)*num_points_on_upper_and_cur_levels);
        for (j = 0; j < num_points_on_upper_and_cur_levels; j++) {
            dci_inst->num_finest_level_points[1][j] = dci_inst->next_level_ranges[1][j].num;
        }
        for (i = 2; i < actual_num_levels; i++) {
            num_points_on_upper_and_cur_levels -= num_points_on_level[i-1];
            dci_inst->num_finest_level_points[i] = (int *)malloc(sizeof(int)*num_points_on_upper_and_cur_levels);
            for (j = 0; j < num_points_on_upper_and_cur_levels; j++) {
                dci_inst->num_finest_level_points[i][j] = 0;
                int k;
                for (k = dci_inst->next_level_ranges[i][j].start; k < dci_inst->next_level_ranges[i][j].start+dci_inst->next_level_ranges[i][j].num; k++) {
                    dci_inst->num_finest_level_points[i][j] += dci_inst->num_finest_level_points[i-1][k];
                }
            }
        }
    }
    
    if (actual_num_levels >= 2) {
        for (i = 0; i < actual_num_levels; i++) {
            free(level_members[i]);
        }
        free(level_members);
        free(assigned_parent);
    }
    free(data_proj);
    
}

static inline int dci_next_closest_proj(const idx_elem* const index, int* const left_pos, int* const right_pos, const double query_proj, const int num_elems) {

    int cur_pos;
    if (*left_pos == -1 && *right_pos == num_elems) {
        cur_pos = -1;
    } else if (*left_pos == -1) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else if (*right_pos == num_elems) {
        cur_pos = *left_pos;
        --(*left_pos);
    } else if (index[*right_pos].key - query_proj < query_proj - index[*left_pos].key) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else {
        cur_pos = *left_pos;
        --(*left_pos);
    }
    return cur_pos;
}

// Returns the index of the element whose key is the largest that is less than the key
// Returns an integer from -1 to num_elems - 1 inclusive
// Could return -1 if all elements are greater or equal to key
static inline int dci_search_index(const idx_elem* const index, const double key, const int num_elems) {
    int start_pos, end_pos, cur_pos;
    
    start_pos = -1;
    end_pos = num_elems - 1;
    cur_pos = (start_pos + end_pos + 2) / 2;
    
    while (start_pos < end_pos) {
        if (index[cur_pos].key < key) {
            start_pos = cur_pos;
        } else {
            end_pos = cur_pos - 1;
        }
        cur_pos = (start_pos + end_pos + 2) / 2;
    }
    
    return start_pos;
}

// Blind querying does not compute distances or look at the values of indexed vectors
// Either num_to_visit or prop_to_visit can be -1; similarly, either num_to_retrieve or prop_to_retrieve can be -1
// Returns whenever we have visited max(num_to_visit, prop_to_visit*num_points) points or retrieved max(num_to_retrieve, prop_to_retrieve*num_points) points, whichever happens first
static int dci_query_single_point_single_level(const dci* const dci_inst, const idx_elem* const indices, int num_points, int num_neighbours, const double* const query, const double* const query_proj, const dci_query_config query_config, const int* const num_finest_level_points, idx_elem* const top_candidates, double* const index_priority, int* const left_pos, int* const right_pos, int* const cur_point_local_ids, int* const cur_point_global_ids, int* const counts, double* const candidate_dists, double* const farthest_dists) {
    
    int i, j, k, m, h, top_h;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    int cur_pos;
    double cur_dist, cur_proj_dist, top_index_priority;
    int num_candidates = 0;
    double last_top_candidate_dist = -1.0;   // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_returned = 0;
    int num_returned_finest_level_points = 0;
    int num_dist_evals = 0;
    
    assert(num_neighbours > 0);
    
    int num_points_to_retrieve = max_i(query_config.num_to_retrieve, (int)ceil(query_config.prop_to_retrieve*num_points));
    int num_projs_to_visit = max_i(query_config.num_to_visit*dci_inst->num_simp_indices, (int)ceil(query_config.prop_to_visit*num_points*dci_inst->num_simp_indices));
    
    for (i = 0; i < dci_inst->num_comp_indices*num_points; i++) {
        counts[i] = 0;
    }
    
    if (!query_config.blind) {
        for (m = 0; m < dci_inst->num_comp_indices; m++) {
            farthest_dists[m] = 0.0;
        }
    }
    for (i = 0; i < num_points; i++) {
        candidate_dists[i] = -1.0;
    }
    
    for (i = 0; i < num_indices; i++) {
        left_pos[i] = dci_search_index(&(indices[i*num_points]), query_proj[i], num_points);
        right_pos[i] = left_pos[i] + 1;
    }
    for (i = 0; i < num_indices; i++) {
        cur_pos = dci_next_closest_proj(&(indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points);
        assert(cur_pos >= 0);    // There should be at least one point in the index
        index_priority[i] = abs_d(indices[cur_pos+i*num_points].key - query_proj[i]);
        cur_point_local_ids[i] = indices[cur_pos+i*num_points].local_value;
        assert(cur_point_local_ids[i] >= 0);
        cur_point_global_ids[i] = indices[cur_pos+i*num_points].global_value;
        assert(cur_point_global_ids[i] >= 0);
    }
    
    k = 0;
    while (k < num_points*dci_inst->num_simp_indices) {
        for (m = 0; m < dci_inst->num_comp_indices; m++) {
            top_index_priority = DBL_MAX;
            top_h = -1;
            for (h = 0; h < dci_inst->num_simp_indices; h++) {
                if (index_priority[h+m*dci_inst->num_simp_indices] < top_index_priority) {
                    top_index_priority = index_priority[h+m*dci_inst->num_simp_indices];
                    top_h = h;
                }
            }
            if (top_h >= 0) {
                i = top_h+m*dci_inst->num_simp_indices;
                counts[cur_point_local_ids[i]+m*num_points]++;
                
                if (counts[cur_point_local_ids[i]+m*num_points] == dci_inst->num_simp_indices) {
                    if (query_config.blind) {
                        if (candidate_dists[cur_point_local_ids[i]] < 0.0) {                   
                            top_candidates[num_candidates].local_value = cur_point_local_ids[i];
                            top_candidates[num_candidates].global_value = cur_point_global_ids[i];
                            candidate_dists[cur_point_local_ids[i]] = top_index_priority;
                            num_candidates++;
                            if (query_config.min_num_finest_level_points > 1) {
                                num_returned_finest_level_points += num_finest_level_points[cur_point_local_ids[i]];
                            }
                        } else if (top_index_priority > candidate_dists[cur_point_local_ids[i]]) {
                            candidate_dists[cur_point_local_ids[i]] = top_index_priority;
                        }
                    } else {
                        if (candidate_dists[cur_point_local_ids[i]] < 0.0) {
                            // Compute distance
                            cur_dist = compute_dist(&(dci_inst->data[((long long int)cur_point_global_ids[i])*dci_inst->dim]), query, dci_inst->dim);
                            candidate_dists[cur_point_local_ids[i]] = cur_dist;
                            num_dist_evals++;
                            
                            if (num_candidates < num_neighbours) {
                                top_candidates[num_returned].key = cur_dist;
                                top_candidates[num_returned].local_value = cur_point_local_ids[i];
                                top_candidates[num_returned].global_value = cur_point_global_ids[i];
                                if (cur_dist > last_top_candidate_dist) {
                                    last_top_candidate_dist = cur_dist;
                                    last_top_candidate = num_returned;
                                }
                                num_returned++;
                                if (query_config.min_num_finest_level_points > 1) {
                                    num_returned_finest_level_points += num_finest_level_points[cur_point_local_ids[i]];
                                }
                            } else if (cur_dist < last_top_candidate_dist) {
                                if (query_config.min_num_finest_level_points > 1 && 
                                num_returned_finest_level_points + num_finest_level_points[cur_point_local_ids[i]] - num_finest_level_points[top_candidates[last_top_candidate].local_value]
                                 < query_config.min_num_finest_level_points) {
                                    // Add
                                    top_candidates[num_returned].key = cur_dist;
                                    top_candidates[num_returned].local_value = cur_point_local_ids[i];
                                    top_candidates[num_returned].global_value = cur_point_global_ids[i];
                                    if (cur_dist > last_top_candidate_dist) {
                                        last_top_candidate_dist = cur_dist;
                                        last_top_candidate = num_returned;
                                    }
                                    num_returned++;
                                    num_returned_finest_level_points += num_finest_level_points[cur_point_local_ids[i]];
                                } else {
                                    // Replace
                                    // If num_returned > num_neighbours, may need to delete, but will leave this to the end
                                    if (query_config.min_num_finest_level_points > 1) {
                                        num_returned_finest_level_points += num_finest_level_points[cur_point_local_ids[i]] - num_finest_level_points[top_candidates[last_top_candidate].local_value];
                                    }
                                    top_candidates[last_top_candidate].key = cur_dist;
                                    top_candidates[last_top_candidate].local_value = cur_point_local_ids[i];
                                    top_candidates[last_top_candidate].global_value = cur_point_global_ids[i];
                                    last_top_candidate_dist = -1.0;
                                    for (j = 0; j < num_returned; j++) {
                                        if (top_candidates[j].key > last_top_candidate_dist) {
                                            last_top_candidate_dist = top_candidates[j].key;
                                            last_top_candidate = j;
                                        }
                                    }
                                }
                            }
                            num_candidates++;
                        } else {
                            cur_dist = candidate_dists[cur_point_local_ids[i]];
                        }
                        if (cur_dist > farthest_dists[m]) {
                            farthest_dists[m] = cur_dist;
                        }
                    }
                }
                
                cur_pos = dci_next_closest_proj(&(indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points);

                if (cur_pos >= 0) {
                    cur_proj_dist = abs_d(indices[cur_pos+i*num_points].key - query_proj[i]);
                    index_priority[i] = cur_proj_dist;
                    cur_point_local_ids[i] = indices[cur_pos+i*num_points].local_value;
                    cur_point_global_ids[i] = indices[cur_pos+i*num_points].global_value;
                } else {
                    index_priority[i] = DBL_MAX;
                    cur_point_local_ids[i] = -1;
                    cur_point_global_ids[i] = -1;
                }
            }
        }
        if (num_candidates >= num_neighbours && num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
            if (k + 1 >= num_projs_to_visit || num_candidates >= num_points_to_retrieve) {
                break;
            }
        }
        k++;
    }
    if (query_config.blind) {
        for (j = 0; j < num_candidates; j++) {
            top_candidates[j].key = candidate_dists[top_candidates[j].local_value];
        }
        qsort(top_candidates, num_candidates, sizeof(idx_elem), dci_compare_idx_elem);        
        num_returned = min_i(num_candidates, num_points_to_retrieve);
    } else {
        qsort(top_candidates, num_returned, sizeof(idx_elem), dci_compare_idx_elem);
        if (query_config.min_num_finest_level_points > 1) {
            num_returned_finest_level_points = 0;
            // Delete the points that are not needed to make num_returned_finest_level_points exceed query_config.min_num_finest_level_points
            for (j = 0; j < num_returned - 1; j++) {
                num_returned_finest_level_points += num_finest_level_points[top_candidates[j].local_value];
                if (num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
                    break;
                }
            }
            num_returned = max_i(min_i(num_neighbours, num_points), j + 1);
        }
    }
    return num_returned;
}

static int dci_query_single_point(const dci* const dci_inst, int num_populated_levels, int num_neighbours, const double* const query, const double* const query_proj, dci_query_config query_config, idx_elem* const top_candidates) {
    
    int i, j, k, l;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    int num_points_to_expand;
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours);
    if (query_config.blind) {
        max_num_points_to_expand += dci_inst->num_comp_indices-1;
    }
    idx_elem points_to_expand[max_num_points_to_expand*max_num_points_to_expand];
    idx_elem points_to_expand_next[max_num_points_to_expand*max_num_points_to_expand];
    
    int top_level_counts[dci_inst->num_comp_indices*dci_inst->num_coarse_points];
    double top_level_candidate_dists[dci_inst->num_coarse_points];
    
    // Only used when non-blind querying is used
    double top_level_farthest_dists[dci_inst->num_comp_indices];
    
    int top_level_left_pos[num_indices];
    int top_level_right_pos[num_indices];
    
    double top_level_index_priority[num_indices];       // Relative priority of simple indices in each composite index
    int top_level_cur_point_local_ids[num_indices];     // Point at the current location in each index
    int top_level_cur_point_global_ids[num_indices];    // Point at the current location in each index
    
    int num_top_candidates[max_num_points_to_expand];
    
    int total_num_top_candidates, num_finest_level_points_to_expand;
    
    assert(num_populated_levels <= dci_inst->num_levels);
    
    if (num_populated_levels <= 1) {
        
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;
        
        num_points_to_expand = dci_query_single_point_single_level(dci_inst, dci_inst->indices[dci_inst->num_levels - 1], dci_inst->num_coarse_points, num_neighbours, query, query_proj, query_config, NULL, points_to_expand_next, top_level_index_priority, top_level_left_pos, top_level_right_pos, top_level_cur_point_local_ids, top_level_cur_point_global_ids, top_level_counts, top_level_candidate_dists, top_level_farthest_dists);
        
    } else {
        
        assert(query_config.field_of_view > 0);
        
        if (query_config.blind) {
            query_config.num_to_retrieve = query_config.field_of_view;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = num_neighbours;
        
        if (num_neighbours > 1) {
            num_points_to_expand = dci_query_single_point_single_level(dci_inst, dci_inst->indices[dci_inst->num_levels - 1], dci_inst->num_coarse_points, query_config.field_of_view, query, query_proj, query_config, dci_inst->num_finest_level_points[dci_inst->num_levels - 1], points_to_expand, top_level_index_priority, top_level_left_pos, top_level_right_pos, top_level_cur_point_local_ids, top_level_cur_point_global_ids, top_level_counts, top_level_candidate_dists, top_level_farthest_dists);
        } else {
            num_points_to_expand = dci_query_single_point_single_level(dci_inst, dci_inst->indices[dci_inst->num_levels - 1], dci_inst->num_coarse_points, query_config.field_of_view, query, query_proj, query_config, NULL, points_to_expand, top_level_index_priority, top_level_left_pos, top_level_right_pos, top_level_cur_point_local_ids, top_level_cur_point_global_ids, top_level_counts, top_level_candidate_dists, top_level_farthest_dists);
        }
        
        for (i = dci_inst->num_levels - 2; i >= dci_inst->num_levels - num_populated_levels + 1; i--) {
        
            #pragma omp parallel for
            for (j = 0; j < num_points_to_expand; j++) {
                range mid_level_indices_range = dci_inst->next_level_ranges[i+1][points_to_expand[j].local_value];
            
                int mid_level_counts[dci_inst->num_comp_indices*mid_level_indices_range.num];
                double mid_level_candidate_dists[mid_level_indices_range.num];
            
                // Only used when non-blind querying is used
                double mid_level_farthest_dists[dci_inst->num_comp_indices];
            
                int num_indices_local = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
            
                int mid_level_left_pos[num_indices_local];
                int mid_level_right_pos[num_indices_local];
            
                double mid_level_index_priority[num_indices_local];       // Relative priority of simple indices in each composite index
                int mid_level_cur_point_local_ids[num_indices_local];     // Point at the current location in each index
                int mid_level_cur_point_global_ids[num_indices_local];    // Point at the current location in each index
                
                int m;
                
                if (num_neighbours > 1) {
                    num_top_candidates[j] = dci_query_single_point_single_level(dci_inst, &(dci_inst->indices[i][mid_level_indices_range.start*num_indices]), mid_level_indices_range.num, query_config.field_of_view, query, query_proj, query_config, &(dci_inst->num_finest_level_points[i][mid_level_indices_range.start]), &(points_to_expand_next[j*max_num_points_to_expand]), mid_level_index_priority, mid_level_left_pos, mid_level_right_pos, mid_level_cur_point_local_ids, mid_level_cur_point_global_ids, mid_level_counts, mid_level_candidate_dists, mid_level_farthest_dists);
                } else {
                    num_top_candidates[j] = dci_query_single_point_single_level(dci_inst, &(dci_inst->indices[i][mid_level_indices_range.start*num_indices]), mid_level_indices_range.num, query_config.field_of_view, query, query_proj, query_config, NULL, &(points_to_expand_next[j*max_num_points_to_expand]), mid_level_index_priority, mid_level_left_pos, mid_level_right_pos, mid_level_cur_point_local_ids, mid_level_cur_point_global_ids, mid_level_counts, mid_level_candidate_dists, mid_level_farthest_dists);
                }
                
                for (m = 0; m < num_top_candidates[j]; m++) {
                    points_to_expand_next[j*max_num_points_to_expand+m].local_value += mid_level_indices_range.start;
                }
                
                assert(num_top_candidates[j] <= max_num_points_to_expand);
            
            }
            
            // Remove empty slots in points_to_expand_next and make it contiguous
            for (k = 0; k < num_points_to_expand; k++) {
                if (num_top_candidates[k] < max_num_points_to_expand) {
                    break;
                }
            }
            if (k < num_points_to_expand) {
                total_num_top_candidates = k*max_num_points_to_expand + num_top_candidates[k];
                k++;
                for (; k < num_points_to_expand; k++) {
                    for (l = 0; l < num_top_candidates[k]; l++) {
                        points_to_expand_next[total_num_top_candidates] = points_to_expand_next[k*max_num_points_to_expand+l];
                        total_num_top_candidates++;
                    }
                }
            } else {
                total_num_top_candidates = num_points_to_expand*max_num_points_to_expand;
            }
            qsort(points_to_expand_next, total_num_top_candidates, sizeof(idx_elem), dci_compare_idx_elem);
            
            if (num_neighbours > 1) {
                num_finest_level_points_to_expand = 0;
                // Delete the points that are not needed to make num_finest_level_points_to_expand exceed num_neighbours
                for (k = 0; k < total_num_top_candidates - 1; k++) {
                    num_finest_level_points_to_expand += dci_inst->num_finest_level_points[i][points_to_expand_next[k].local_value];
                    if (num_finest_level_points_to_expand >= num_neighbours) {
                        break;
                    }
                }
                
                num_points_to_expand = max_i(min_i(query_config.field_of_view, total_num_top_candidates), k + 1);
                
            } else {
                num_points_to_expand = min_i(query_config.field_of_view, total_num_top_candidates);
            }
            
            for (k = 0; k < num_points_to_expand; k++) {
                points_to_expand[k] = points_to_expand_next[k];
            }
        }
    
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;
        
        #pragma omp parallel for
        for (j = 0; j < num_points_to_expand; j++) {
            range bottom_level_indices_range = dci_inst->next_level_ranges[dci_inst->num_levels - num_populated_levels + 1][points_to_expand[j].local_value];
            
            int bottom_level_counts[dci_inst->num_comp_indices*bottom_level_indices_range.num];
            double bottom_level_candidate_dists[bottom_level_indices_range.num];
        
            // Only used when non-blind querying is used
            double bottom_level_farthest_dists[dci_inst->num_comp_indices];
        
            int num_indices_local = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
        
            int bottom_level_left_pos[num_indices_local];
            int bottom_level_right_pos[num_indices_local];
            
            double bottom_level_index_priority[num_indices_local];       // Relative priority of simple indices in each composite index
            int bottom_level_cur_point_local_ids[num_indices_local];     // Point at the current location in each index
            int bottom_level_cur_point_global_ids[num_indices_local];    // Point at the current location in each index
        
            int m;
            
            num_top_candidates[j] = dci_query_single_point_single_level(dci_inst, &(dci_inst->indices[dci_inst->num_levels - num_populated_levels][bottom_level_indices_range.start*num_indices]), bottom_level_indices_range.num, num_neighbours, query, query_proj, query_config, NULL, &(points_to_expand_next[j*num_neighbours]), bottom_level_index_priority, bottom_level_left_pos, bottom_level_right_pos, bottom_level_cur_point_local_ids, bottom_level_cur_point_global_ids, bottom_level_counts, bottom_level_candidate_dists, bottom_level_farthest_dists);
            
            for (m = 0; m < num_top_candidates[j]; m++) {
                points_to_expand_next[j*num_neighbours+m].local_value += bottom_level_indices_range.start;
            }
            
            assert(num_top_candidates[j] <= num_neighbours);
        }
        
        // Remove empty slots in points_to_expand_next and make it contiguous
        for (k = 0; k < num_points_to_expand; k++) {
            if (num_top_candidates[k] < num_neighbours) {
                break;
            }
        }
        if (k < num_points_to_expand) {
            total_num_top_candidates = k*num_neighbours + num_top_candidates[k];
            k++;
            for (; k < num_points_to_expand; k++) {
                for (l = 0; l < num_top_candidates[k]; l++) {
                    points_to_expand_next[total_num_top_candidates] = points_to_expand_next[k*num_neighbours+l];
                    total_num_top_candidates++;
                }
            }
        } else {
            total_num_top_candidates = num_points_to_expand*num_neighbours;
        }
        
        qsort(points_to_expand_next, total_num_top_candidates, sizeof(idx_elem), dci_compare_idx_elem);
        
        num_points_to_expand = min_i(num_neighbours, total_num_top_candidates);
        
    }
    for (k = 0; k < num_points_to_expand; k++) {
        top_candidates[k] = points_to_expand_next[k];
    }
    
    return num_points_to_expand;
    
}

static void dci_assign_parent(dci* const dci_inst, const int num_populated_levels, const int num_queries, const int *selected_query_pos, const double* const query, const double* const query_proj, const dci_query_config query_config, tree_node* const assigned_parent) {
    
    int j;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    
    #pragma omp parallel for
    for (j = 0; j < num_queries; j++) {
        int cur_num_returned;
        idx_elem top_candidate;
        
        cur_num_returned = dci_query_single_point(dci_inst, num_populated_levels, 1, &(query[((long long int)selected_query_pos[j])*dci_inst->dim]), &(query_proj[selected_query_pos[j]*num_indices]), query_config, &top_candidate);
        assert(cur_num_returned == 1);
        
        assigned_parent[j].parent = top_candidate.local_value;
        assigned_parent[j].child = selected_query_pos[j];
        
    }
}

// nearest_neighbour_dists can be NULL
// num_returned can be NULL; if not NULL, it is populated with the number of returned points for each query - it should be of size num_queries
// CAUTION: This function allocates memory for each nearest_neighbours[j], nearest_neighbour_dists[j], so we need to deallocate them outside of this function!
void dci_query(dci* const dci_inst, const int dim, const int num_queries, const double* const query, const int num_neighbours, const dci_query_config query_config, int** const nearest_neighbours, double** const nearest_neighbour_dists, int* const num_returned) {
    
    int j;
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    
    double* query_proj;
    
    assert(dim == dci_inst->dim);
    assert(num_neighbours > 0);
    
    query_proj = (double *)memalign(64, sizeof(double)*num_indices*num_queries);
    matmul(num_indices, num_queries, dim, dci_inst->proj_vec, query, query_proj);
    
    #pragma omp parallel for
    for (j = 0; j < num_queries; j++) {
        
        int k;
        int cur_num_returned;
        
        idx_elem top_candidates[num_neighbours];        // Maintains the top-k candidates
        
        cur_num_returned = dci_query_single_point(dci_inst, dci_inst->num_levels, num_neighbours, &(query[j*dim]), &(query_proj[j*num_indices]), query_config, top_candidates);
        
        assert(cur_num_returned <= num_neighbours);
        nearest_neighbours[j] = (int *)malloc(sizeof(int) * cur_num_returned);
        for (k = 0; k < cur_num_returned; k++) {
            nearest_neighbours[j][k] = top_candidates[k].global_value;
        }
        if (nearest_neighbour_dists) {
            nearest_neighbour_dists[j] = (double *)malloc(sizeof(double) * cur_num_returned);
            for (k = 0; k < cur_num_returned; k++) {
                nearest_neighbour_dists[j][k] = top_candidates[k].key;
            }
        }
        if (num_returned) {
            num_returned[j] = cur_num_returned;
        }
    }
    free(query_proj);
    
}

void dci_clear(dci* const dci_inst) {
    int i;
    if (dci_inst->indices) {
        for (i = 0; i < dci_inst->num_levels; i++) {
            free(dci_inst->indices[i]);
        }
        free(dci_inst->indices);
        dci_inst->indices = NULL;
    }
    if (dci_inst->next_level_ranges) {
        for (i = 1; i < dci_inst->num_levels; i++) {
            free(dci_inst->next_level_ranges[i]);
        }
        free(dci_inst->next_level_ranges);
        dci_inst->next_level_ranges = NULL;
    }
    if (dci_inst->num_finest_level_points) {
        for (i = 1; i < dci_inst->num_levels; i++) {
            free(dci_inst->num_finest_level_points[i]);
        }
        free(dci_inst->num_finest_level_points);
        dci_inst->num_finest_level_points = NULL;
    }
    dci_inst->data = NULL;
    dci_inst->num_points = 0;
    dci_inst->num_levels = 0;
    dci_inst->num_coarse_points = 0;
}

void dci_reset(dci* const dci_inst) {
    srand48(time(NULL));
    dci_clear(dci_inst);
    dci_gen_proj_vec(dci_inst->proj_vec, dci_inst->dim, dci_inst->num_comp_indices*dci_inst->num_simp_indices);
} 

void dci_free(const dci* const dci_inst) {
    int i;
    if (dci_inst->indices) {
        for (i = 0; i < dci_inst->num_levels; i++) {
            free(dci_inst->indices[i]);
        }
        free(dci_inst->indices);
    }
    if (dci_inst->next_level_ranges) {
        for (i = 1; i < dci_inst->num_levels; i++) {
            free(dci_inst->next_level_ranges[i]);
        }
        free(dci_inst->next_level_ranges);
    }
    if (dci_inst->num_finest_level_points) {
        for (i = 1; i < dci_inst->num_levels; i++) {
            free(dci_inst->num_finest_level_points[i]);
        }
        free(dci_inst->num_finest_level_points);
    }
    free(dci_inst->proj_vec);
}
