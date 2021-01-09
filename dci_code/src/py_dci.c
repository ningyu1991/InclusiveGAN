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

#include "Python.h"
#include "numpy/arrayobject.h"
#include "dci.h"

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifdef PY3K

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

#endif

// DCI struct with some additional structures for Python-specific bookkeeping
typedef struct py_dci {
    dci dci_inst;
    PyArrayObject *py_array;
    int data_idx_offset;
} py_dci;

// Called automatically by the garbage collector
static void py_dci_free(PyObject *py_dci_inst_wrapper) {
    
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }
    
    dci_free(&(py_dci_inst->dci_inst));
    
    free(py_dci_inst);
}

static PyObject *py_dci_new(PyObject *self, PyObject *args) {
    
    int dim, num_comp_indices, num_simp_indices;
    
    if (!PyArg_ParseTuple(args, "iii", &dim, &num_comp_indices, &num_simp_indices)) return NULL;
    
    py_dci *py_dci_inst = (py_dci *)malloc(sizeof(py_dci));
        
    dci_init(&(py_dci_inst->dci_inst), dim, num_comp_indices, num_simp_indices);
    
    py_dci_inst->py_array = NULL;
    py_dci_inst->data_idx_offset = 0;
    
    // Returns new reference
    PyObject *py_dci_inst_wrapper = PyCapsule_New(py_dci_inst, "py_dci_inst", py_dci_free);
    
    return py_dci_inst_wrapper;
}

// Borrows *py_dci_inst_wrapper, py_dci_inst owns at most one copy of *py_data
static PyObject *py_dci_add(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data;
    int dim, start_idx, end_idx, num_levels, num_to_visit, num_to_retrieve, field_of_view, num_new_points;
    double prop_to_visit, prop_to_retrieve;
    unsigned char blind;
    dci_query_config construction_query_config;
    py_dci *py_dci_inst;
    double *data;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!iiibiiddi", &py_dci_inst_wrapper, &PyArray_Type, &py_data, &start_idx, &end_idx, &num_levels, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_data) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (double *)py_data->data;
	num_new_points = end_idx - start_idx;
	dim = py_data->dimensions[1];
	
    if (num_new_points > 0) {
        
        construction_query_config.blind = blind;
        construction_query_config.num_to_visit = num_to_visit;
        construction_query_config.num_to_retrieve = num_to_retrieve;
        construction_query_config.prop_to_visit = prop_to_visit;
        construction_query_config.prop_to_retrieve = prop_to_retrieve;
        construction_query_config.field_of_view = field_of_view;
        
        dci_add(&(py_dci_inst->dci_inst), dim, num_new_points, &(data[start_idx*dim]), num_levels, construction_query_config);
        py_dci_inst->data_idx_offset = start_idx;
        py_dci_inst->py_array = py_data;
        
        // py_dci_inst owns a reference to py_data and relinquishes it when database is cleared
        Py_INCREF(py_data);
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_query(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_query, *py_nearest_neighbour_idx, *py_nearest_neighbour_dists, *py_num_returned;
    int i, j, k, dim, num_neighbours, num_to_visit, num_to_retrieve, num_queries, field_of_view;
    unsigned char blind;
    double prop_to_visit, prop_to_retrieve;
    py_dci *py_dci_inst;
    double *query, *nearest_neighbour_dists_flattened;
    int *nearest_neighbour_idx, *num_returned;
    dci_query_config query_config;
    int **nearest_neighbours;
    double **nearest_neighbour_dists;
    npy_intp py_nearest_neighbours_shape[1];
    npy_intp py_num_returned_shape[1];
    
    if (!PyArg_ParseTuple(args, "OO!ibiiddi", &py_dci_inst_wrapper, &PyArray_Type, &py_query, &num_neighbours, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_query) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    query = (double *)py_query->data;
	num_queries = py_query->dimensions[0];
	dim = py_query->dimensions[1];
        
    py_num_returned_shape[0] = num_queries;
    
    py_num_returned = (PyArrayObject *)PyArray_SimpleNew(1, py_num_returned_shape, NPY_INT);
    num_returned = (int *)py_num_returned->data;
    
    query_config.blind = blind;
    query_config.num_to_visit = num_to_visit;
    query_config.num_to_retrieve = num_to_retrieve;
    query_config.prop_to_visit = prop_to_visit;
    query_config.prop_to_retrieve = prop_to_retrieve;
    query_config.field_of_view = field_of_view;
    
    nearest_neighbours = (int **)malloc(sizeof(int *)*num_queries);
    nearest_neighbour_dists = (double **)malloc(sizeof(double *)*num_queries);
    
    dci_query(&(py_dci_inst->dci_inst), dim, num_queries, query, num_neighbours, query_config, nearest_neighbours, nearest_neighbour_dists, num_returned);

    py_nearest_neighbours_shape[0] = 0;
    for (i = 0; i < num_queries; i++) {
        py_nearest_neighbours_shape[0] += num_returned[i];
    }
    
    py_nearest_neighbour_idx = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_INT);
    nearest_neighbour_idx = (int *)py_nearest_neighbour_idx->data;
    
    k = 0;
    for (i = 0; i < num_queries; i++) {
        for (j = 0; j < num_returned[i]; j++) {
            nearest_neighbour_idx[k] = nearest_neighbours[i][j] + py_dci_inst->data_idx_offset;
            k++;
        }
    }
    
    // Assuming row-major layout, matrix is of size num_queries x num_neighbours
    py_nearest_neighbour_dists = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_DOUBLE);
    nearest_neighbour_dists_flattened = (double *)py_nearest_neighbour_dists->data;
    k = 0;
    for (i = 0; i < num_queries; i++) {
        for (j = 0; j < num_returned[i]; j++) {
            nearest_neighbour_dists_flattened[k] = nearest_neighbour_dists[i][j];
            k++;
        }
    }
    
    for (i = 0; i < num_queries; i++) {
        free(nearest_neighbours[i]);
    }
    free(nearest_neighbours);
    for (i = 0; i < num_queries; i++) {
        free(nearest_neighbour_dists[i]);
    }
    free(nearest_neighbour_dists);
    
    return Py_BuildValue("NNN", py_nearest_neighbour_idx, py_nearest_neighbour_dists, py_num_returned);
}

// Borrows *py_dci_inst_wrapper, relinquishes *py_data
static PyObject *py_dci_clear(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
	
    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }
    
    dci_clear(&(py_dci_inst->dci_inst));
    py_dci_inst->py_array = NULL;
    py_dci_inst->data_idx_offset = 0;
    
    Py_INCREF(Py_None);
    return Py_None;
    
}

// Borrows *py_dci_inst_wrapper, relinquishes *py_data
static PyObject *py_dci_reset(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }
    
    dci_reset(&(py_dci_inst->dci_inst));
    py_dci_inst->py_array = NULL;
    py_dci_inst->data_idx_offset = 0;
    
    Py_INCREF(Py_None);
    return Py_None;
    
}

static PyObject *py_dci_get_num_points(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
	return Py_BuildValue("i", (py_dci_inst->dci_inst).num_points);
}

static PyObject *py_dci_get_num_levels(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
	return Py_BuildValue("i", (py_dci_inst->dci_inst).num_levels);
}

static PyObject *py_dci_get_proj_vec(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    PyArrayObject *py_proj_vec;
    npy_intp py_proj_vec_shape[2];
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    py_proj_vec_shape[0] = (py_dci_inst->dci_inst).num_comp_indices*(py_dci_inst->dci_inst).num_simp_indices;
    py_proj_vec_shape[1] = (py_dci_inst->dci_inst).dim;
    // Assuming row-major layout, matrix is of size (num_comp_indices*num_simp_indices) x dim
    py_proj_vec = (PyArrayObject *)PyArray_SimpleNewFromData(2, py_proj_vec_shape, NPY_DOUBLE, (py_dci_inst->dci_inst).proj_vec);
    // py_proj_vec owns a reference to py_dci_inst_wrapper
    py_proj_vec->base = py_dci_inst_wrapper;
    Py_INCREF(py_dci_inst_wrapper);
    
    return (PyObject *)py_proj_vec;
}

// Methods table - maps names in Python to C functions  
static PyMethodDef py_dci_module_methods[] = {
    {"new", py_dci_new, METH_VARARGS, "Create new DCI instance."},
    {"add", py_dci_add, METH_VARARGS, "Add data."},
    {"query", py_dci_query, METH_VARARGS, "Search for nearest neighbours."},
    {"clear", py_dci_clear, METH_VARARGS, "Delete all data."},
    {"reset", py_dci_reset, METH_VARARGS, "Delete all data and regenerate projection directions."},
    {"get_num_points", py_dci_get_num_points, METH_VARARGS, "Get the number of points indexed by DCI instance. "},
    {"get_num_levels", py_dci_get_num_levels, METH_VARARGS, "Get the number of levels in DCI instance. "},
    {"get_proj_vec", py_dci_get_proj_vec, METH_VARARGS, "Get the projection vectors used by DCI instance. "},
    {NULL, NULL, 0, NULL}
};

#ifdef PY3K

static int py_dci_module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int py_dci_module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef py_dci_module_def = {
        PyModuleDef_HEAD_INIT,
        "_dci",
        NULL,
        sizeof(struct module_state),
        py_dci_module_methods,
        NULL,
        py_dci_module_traverse,
        py_dci_module_clear,
        NULL
};

// Module name is "_dci"
PyMODINIT_FUNC PyInit__dci(void) {
    PyObject *module = PyModule_Create(&py_dci_module_def);
    import_array();     // Import Numpy
    return module;
}

#else

// Module name is "_dci"
PyMODINIT_FUNC init_dci(void) {
    (void) Py_InitModule("_dci", py_dci_module_methods);
    import_array();     // Import Numpy
}

#endif
