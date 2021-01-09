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

#ifndef UTIL_H
#define UTIL_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_MKL
#define DGEMM dgemm
#else
#define DGEMM dgemm_
#endif  // USE_MKL

// BLAS native Fortran interface
extern void DGEMM(const char* const transa, const char* const transb, const int* const m, const int* const n, const int* const k, const double* const alpha, const double* const A, const int* const lda, const double* const B, const int* const ldb, const double* const beta, double* const C, const int* const ldc);

void matmul(const int M, const int N, const int K, const double* const A, const double* const B, double* const C);

void gen_data(double* const data, const int ambient_dim, const int intrinsic_dim, const int num_points);

double compute_dist(const double* const vec1, const double* const vec2, const int dim);

double rand_normal();

void print_matrix(const double* const data, const int num_rows, const int num_cols);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
