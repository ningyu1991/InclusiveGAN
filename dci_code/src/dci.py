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
import _dci

class ProtectedArray(object):
    # when_readable is a function that returns True when reading is allowed
    def __init__(self, base_array, when_readable = None, read_error = None, when_writable = None, write_error = None):
        self._base = base_array
        self._when_readable = when_readable
        self._read_error = read_error
        self._when_writable = when_writable
        self._write_error = write_error
        
    def __getitem__(self, indices):
        if self._when_readable is not None and not self._when_readable(indices):
            if self._read_error is None:
               raise RuntimeError("array is not currently readable")
            else:
                raise self._read_error(indices)
        return self._base.__getitem__(indices)
        
    def __setitem__(self, indices, value):
        if self._when_writable is not None and not self._when_writable(indices):
            if self._write_error is None:
               raise RuntimeError("array is not currently writable")
            else:
                raise self._write_error(indices)
        self._base.__setitem__(indices, value)
        
    def __getattr__(self, attr):
        return getattr(self._base, attr)
    
    def __repr__(self):
        return repr(self._base)

class DCI(object):
    
    def __init__(self, dim, num_comp_indices = 2, num_simp_indices = 7):
        
        self._dim = dim
        self._num_comp_indices = num_comp_indices
        self._num_simp_indices = num_simp_indices
        self._dci_inst = _dci.new(dim, num_comp_indices, num_simp_indices)
        self._proj_vec = _dci.get_proj_vec(self._dci_inst)
        self._array = None
        self._orig_indices = None    # Used only when the data is originally discontiguous - translates the indices from the contiguous subset of data to the original indices
        
    @property
    def dim(self):
        return self._dim
        
    @property
    def num_comp_indices(self):
        return self._num_comp_indices
        
    @property
    def num_simp_indices(self):
        return self._num_simp_indices
    
    @property
    def num_points(self):
        return _dci.get_num_points(self._dci_inst)
    
    @property
    def num_levels(self):
        return _dci.get_num_levels(self._dci_inst)
    
    @property
    def proj_vec(self):
        return ProtectedArray(self._proj_vec, when_writable = lambda _: self.num_points == 0, write_error = lambda _: AttributeError("can only set projection vectors when the database is empty"))
    
    @proj_vec.setter
    def proj_vec(self, new_proj_vec):
        if self.num_points != 0:
            raise AttributeError("can only set projection vectors when the database is empty")
        # Disallow broadcasting when assigning to proj_vec directly
        new_proj_vec = np.asarray(new_proj_vec)
        if new_proj_vec.shape != self._proj_vec.shape:
            raise ValueError("mismatch between the expected shape of projection vectors (%s) and the supplied shape (%s)" % (repr(self._proj_vec.shape),repr(new_proj_vec.shape)))
        self._proj_vec[...] = new_proj_vec
            
    def _ensure_positive_integer(self, x):
        if not isinstance(x, int):
            raise TypeError("number must be an integer")
        elif x <= 0:
             raise ValueError("number must be positive")
    
    def _check_array(self, arr):
        if arr.shape[1] != self.dim:
            raise ValueError("mismatch between array dimension (%d) and the declared dimension of this DCI instance (%d)" % (arr.shape[1],self.dim))
        if arr.dtype != np.float:
            raise TypeError("array must consist of double-precision floats")
        if not arr.flags.c_contiguous:
            raise ValueError("the memory layout of array must be in row-major (C-order)")
    
    def _check_and_fix_array(self, arr):
        if arr.shape[1] != self.dim:
            raise ValueError("mismatch between array dimension (%d) and the declared dimension of this DCI instance (%d)" % (arr.shape[1],self.dim))
        if arr.dtype == np.float and arr.flags.c_contiguous:
            return arr
        else:
            return np.array(arr, dtype=np.float, copy=False, order='C')
    
    def _check_is_base_array(self, arr):
        # arr cannot be derived from some other array (except if it's just transposed, in which case the data pointer stays the same)
        if arr.base is not None:
            try:
                arr_addr = arr.data
                base = arr
                while base.base is not None:
                    base = base.base
            except AttributeError:
                arr_addr = None
            if arr_addr is None or arr_addr != base.data:
                raise ValueError("array must not be derived from another array, except via the transpose operator. Pass in the original array and specify the indices or make a copy of the derived array.")
    
    def _check_data(self, data):
        self._check_array(data)
        self._check_is_base_array(data)
    
    def _check_and_fix_indices(self, data, indices):
        check_indices_within_bounds = False
        if indices is None:
            is_contiguous = True
            selected_idx = (0,data.shape[0])
        elif isinstance(indices, slice):
            step = indices.step
            start = indices.start
            stop = indices.stop
            if start is None:
                start = 0
            if step is None:
                step = 1
            if start < 0:
                start = data.shape[0] + start
            if stop < 0:
                stop = data.shape[0] + stop
            start = max(start, 0)
            stop = min(stop, data.shape[0])
            if step == 1:
                is_contiguous = True
                selected_idx = (start,stop)
            else:
                is_contiguous = False
                selected_idx = np.arange(start,stop,step,dtype=np.intc)
        elif isinstance(indices, int):
            if indices < 0:
                cur_idx = data.shape[0] + indices
            else:
                cur_idx = indices
            if cur_idx < 0 or cur_idx >= data.shape[0]:
                raise IndexError("index out of bounds")
            is_contiguous = True
            selected_idx = (cur_idx,cur_idx+1)
        elif isinstance(indices, np.ndarray):
            is_contiguous = False
            if indices.ndim == 1:
                if indices.dtype == np.intc:
                    selected_idx = indices
                    if np.any(selected_idx < 0):
                        selected_idx = np.copy(selected_idx)
                        selected_idx[selected_idx < 0] += data.shape[0]
                    check_indices_within_bounds = True
                elif indices.dtype == np.bool:
                    if indices.shape[0] == data.shape[0]:
                        selected_idx = np.nonzero(indices)[0].astype(np.intc)
                    else:
                        raise IndexError("mismatch between the number of boolean indices (%d) and array dimension (%d)" % (indices.shape[0],data.shape[0]))
                elif indices.dtype.kind in np.typecodes['AllInteger']:  # Check if dtype is an integer type; also returns true if dtype is bool                
                    selected_idx = indices.astype(np.intc)
                    selected_idx[selected_idx < 0] += data.shape[0]
                    check_indices_within_bounds = True
                else:
                    raise TypeError("indices must be integers or booleans")
            else:
                raise IndexError("indices must be in an one-dimensional array")
        elif isinstance(indices, list):
            is_contiguous = False
            if isinstance(indices[0], bool):
                selected_idx = np.nonzero(indices)[0].astype(np.intc)
            elif isinstance(indices[0], int):   # Also returns true if indices[0] is bool
                selected_idx = np.array(indices,dtype=np.intc)
                selected_idx[selected_idx < 0] += data.shape[0]
                check_indices_within_bounds = True
            elif isinstance(indices[0], list):
                raise IndexError("indices must be in an one-dimensional array")
            else:
                raise TypeError("indices must be integers or booleans")
        else:
            raise TypeError("indices must be None, a slice object, an integer, an array or list of integers")
            
        if check_indices_within_bounds:
            if np.any(selected_idx < 0) or np.any(selected_idx >= data.shape[0]):
                raise IndexError("some indices (e.g. %d) out of bounds" % (indices[(selected_idx < 0) | (selected_idx >= data.shape[0])][0]))
        
        return is_contiguous,selected_idx
        
    # Indices can be None, a slice object, an integer, an array or list of integers - best to use np.intc type
    def add(self, data, indices = None, num_levels = 2, field_of_view = 10, blind = False, num_to_visit = -1, num_to_retrieve = -1, prop_to_visit = -1.0, prop_to_retrieve = -1.0):
        
        num_points = self.num_points
        
        if num_points > 0:
            raise RuntimeError("DCI class does not support insertion of more than one array. Must combine all arrays into one array before inserting")
        
        if num_levels >= 3:
            self._ensure_positive_integer(field_of_view)
        else:
            field_of_view = -1
        
        if num_to_visit > num_points:
            num_to_visit = num_points
        
        if prop_to_visit < 0.0:
            if num_to_visit < 0:
                prop_to_visit = 1.0
            else:
                prop_to_visit = -1.0 
        else:
            if prop_to_visit > 1.0:
                prop_to_visit = 1.0
        
        if num_to_retrieve > num_points:
            num_to_retrieve = num_points
        
        if prop_to_retrieve < 0.0:
            if num_to_retrieve < 0:
                prop_to_retrieve = 0.002
            else:
                prop_to_retrieve = -1.0
        else:
            if prop_to_retrieve > 1.0:
                prop_to_retrieve = 1.0
        
        self._check_data(data)
        is_contiguous, _indices = self._check_and_fix_indices(data, indices)
        
        if is_contiguous:
            _dci.add(self._dci_inst, data, _indices[0], _indices[1], num_levels, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view)
        else:
            selected_data = data[_indices]
            _dci.add(self._dci_inst, selected_data, 0, _indices.shape[0], num_levels, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view)
            self._orig_indices = _indices
        
        self._array = data
    
    # query is num_queries x dim
    def query(self, query, num_neighbours = -1, field_of_view = 100, blind = False, num_to_visit = -1, num_to_retrieve = -1, prop_to_visit = -1.0, prop_to_retrieve = -1.0):
        _query = self._check_and_fix_array(query)
        
        num_points = self.num_points
        
        if num_neighbours < 0:
            num_neighbours = num_points
        
        self._ensure_positive_integer(num_neighbours)
        
        if self.num_levels >= 2:
            self._ensure_positive_integer(field_of_view)
        else:
            field_of_view = -1
        
        if num_to_visit > num_points:
            num_to_visit = num_points
        
        if prop_to_visit < 0.0:
            if num_to_visit < 0:
                prop_to_visit = 1.0
            else:
                prop_to_visit = -1.0 
        else:
            if prop_to_visit > 1.0:
                prop_to_visit = 1.0
        
        if num_to_retrieve > num_points:
            num_to_retrieve = num_points
        
        if prop_to_retrieve < 0.0:
            if num_to_retrieve < 0:
                prop_to_retrieve = 0.05
            else:
                prop_to_retrieve = -1.0
        else:
            if prop_to_retrieve > 1.0:
                prop_to_retrieve = 1.0
        
        # num_queries x num_neighbours
        _nearest_neighbour_idx, _nearest_neighbour_dists, num_candidates = _dci.query(self._dci_inst, _query, num_neighbours, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view)
        
        if self._orig_indices is not None:
            _nearest_neighbour_idx = self._orig_indices[_nearest_neighbour_idx]
        
        nearest_neighbour_idx = []
        j = 0
        for i in range(_query.shape[0]):
            nearest_neighbour_idx.append(_nearest_neighbour_idx[j:j+num_candidates[i]])
            j += num_candidates[i]
        
        nearest_neighbour_dists = []
        j = 0
        for i in range(_query.shape[0]):
            nearest_neighbour_dists.append(_nearest_neighbour_dists[j:j+num_candidates[i]])
            j += num_candidates[i]
        
        return nearest_neighbour_idx, nearest_neighbour_dists
    
    def clear(self):
        _dci.clear(self._dci_inst)
        self._array = None
        self._orig_indices = None
    
    def reset(self):
        _dci.reset(self._dci_inst)
        self._array = None
        self._orig_indices = None
