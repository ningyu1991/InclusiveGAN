# Code for Fast k-Nearest Neighbour Search via Prioritized DCI
#
# This code implements the method described in the Prioritized DCI paper, which
# can be found at https://arxiv.org/abs/1703.00440
#
# Copyright (C) 2017    Ke Li
#
#
# This file is part of the Dynamic Continuous Indexing reference implementation.
#
# The Dynamic Continuous Indexing reference implementation is free software:
# you can redistribute it and/or modify it under the terms of the GNU Affero
# General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# The Dynamic Continuous Indexing reference implementation is distributed in
# the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with the Dynamic Continuous Indexing reference implementation.  If
# not, see <http://www.gnu.org/licenses/>.
#
# Prerequisites:
# 1. Python 2.7+ or Python 3.1+
# 2. A BLAS library (supported implementations include the reference implementation, ATLAS, OpenBLAS and MKL)
# 3. Python development headers (can be installed using "apt-get install python2.7-dev" or "apt-get install python3.5-dev")
# 4. (If Python interface is desired) Numpy installation
# 5. (If TensorFlow op is desired) TensorFlow installation

# Build Instructions:
# 1.  Set the BLAS variable to "netlib" (for the reference implementation, which will be referred to as Netlib), "atlas" (for ATLAS), 
#       "openblas" (for OpenBLAS) or "mkl" (for MKL). OpenBLAS and MKL are the fastest in our experience.  
# 2.  Set NETLIB_DIR, ATLAS_DIR, OPENBLAS_DIR or MKL_DIR to the directory for your BLAS installation
#       To find the directory, you can consult the output of:
#           "python3 -c 'import numpy.distutils.system_info as sysinfo; print([sysinfo.get_info(s) for s in ["blas", "atlas", "openblas", "mkl"]])'"
#       For Netlib, ATLAS or OpenBLAS, you need to specify the path to the lib directory
#       For MKL, you need to specify the path to the directory *containing* the lib directory
# 3a. If binary executable is desired:
#       Run "make c"
# 3b. If Python interface is desired:
#       First make sure running "python" invokes the Python installation that you intend to use. In particular, if you intend to use
#       Python 3, sometimes you need to invoke it using "python3". In this case, make sure to replace all invocations of "python" 
#       with "python3". 
#       Set PYTHON_DIR to the output of "python3 -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())'" and 
#           NUMPY_DIR to the output of "python3 -c 'import numpy as np; print(np.get_include())'"
#       Run "make py"
#       If the compiler cannot find Python.h, it means the Python development headers are not installed. If you believe they
#       are installed, make sure PYTHON_DIR corresponds to the Python installation that you intend to use. (Sometimes there are
#       multiple Python installations on your system)
# 3c. If TensorFlow op is desired:
#       Set TENSORFLOW_LIB_DIR to the output of "python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'" and 
#           TENSORFLOW_INCL_DIR to the output of "python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'"
#       Run "make tf"
#     If the compiler complains about not being able to find libtensorflow.so or libtensorflow_framework.so, download the 
#     libtensorflow*.tar.gz file for your platform and version of TensorFlow from https://www.tensorflow.org/install/lang_c
#     and copy libtensorflow.so or libtensorflow_framework.so to TENSORFLOW_LIB_DIR


CC=gcc
CPPC=g++
BLAS=netlib
NETLIB_DIR=/usr/lib
ATLAS_DIR=/usr/lib/atlas-base
OPENBLAS_DIR=/usr/lib/openblas-base
MKL_DIR=/opt/common/intel/mkl/10.1.2.024
PYTHON_DIR=/BS/ningyu/work/env/anaconda3/anaconda3/include/python3.6m
NUMPY_DIR=/BS/ningyu/work/env/tensorflow_1.14_cuda_10.0/lib/python3.6/site-packages/numpy/core/include
TENSORFLOW_LIB_DIR=/BS/ningyu/work/env/tensorflow_1.14_cuda_10.0/lib/python3.6/site-packages/tensorflow
TENSORFLOW_INCL_DIR=/BS/ningyu/work/env/tensorflow_1.14_cuda_10.0/lib/python3.6/site-packages/tensorflow/include
TENSORFLOW_INCL_EXT_DIR=/BS/ningyu/work/env/tensorflow_1.14_cuda_10.0/lib/python3.6/site-packages/tensorflow/include/external/nsync/public

SRC_DIR=src
INCL_DIR=include
BUILD_DIR=build
C_BUILD_DIR=$(BUILD_DIR)/c
PY_BUILD_DIR=$(BUILD_DIR)/py
TF_BUILD_DIR=$(BUILD_DIR)/tf

GEN_FLAGS=-Wall -O3 -std=gnu99 -m64 -fopenmp -flto
LIB_FLAGS=-lm

TF_GEN_FLAGS=-Wall -O3 -std=c++11 -m64 -fopenmp -flto -D_GLIBCXX_USE_CXX11_ABI=0
TF_LIB_FLAGS=-L$(TENSORFLOW_LIB_DIR) -ltensorflow_framework -lpthread -ldl -ltensorflow -fPIC

OBJ_FILES=dci.o util.o
INCL_FILES=dci.h util.h

C_OBJ_FILES=$(OBJ_FILES)
TF_OBJ_FILES=$(OBJ_FILES)
PY_OBJ_FILES=py_dci.o $(OBJ_FILES)

ALL_INCL_DIRS=$(INCL_DIR)

ifeq ($(BLAS), netlib)
    LIB_FLAGS += -L$(NETLIB_DIR) -Wl,-rpath $(NETLIB_DIR) -lblas
endif
ifeq ($(BLAS), atlas)
    LIB_FLAGS += -L$(ATLAS_DIR) -Wl,-rpath $(ATLAS_DIR) -latlas
endif
ifeq ($(BLAS), openblas)
    LIB_FLAGS += -L$(OPENBLAS_DIR) -Wl,-rpath $(OPENBLAS_DIR) -lopenblas
endif
ifeq ($(BLAS), mkl)
    ALL_INCL_DIRS += $(MKL_DIR)/include
    GEN_FLAGS += -DUSE_MKL
    LIB_FLAGS += -L$(MKL_DIR)/lib/64 -Wl,-rpath $(MKL_DIR)/lib/64 -lmkl_rt -lpthread -ldl
endif

C_OBJ_PATHS=$(addprefix $(C_BUILD_DIR)/,$(C_OBJ_FILES))
C_INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
C_ALL_INCL_DIRS=$(ALL_INCL_DIRS)
C_ALL_INCL_FLAGS=$(addprefix -I,$(C_ALL_INCL_DIRS))

TF_OBJ_PATHS=$(addprefix $(TF_BUILD_DIR)/,$(TF_OBJ_FILES))
TF_INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
TF_ALL_INCL_DIRS=$(ALL_INCL_DIRS) $(TENSORFLOW_INCL_DIR) $(TENSORFLOW_INCL_EXT_DIR)
TF_ALL_INCL_FLAGS=$(addprefix -I,$(TF_ALL_INCL_DIRS))

PY_OBJ_PATHS=$(addprefix $(PY_BUILD_DIR)/,$(PY_OBJ_FILES))
PY_INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
PY_ALL_INCL_DIRS=$(PYTHON_DIR) $(NUMPY_DIR) $(ALL_INCL_DIRS)
PY_ALL_INCL_FLAGS=$(addprefix -I,$(PY_ALL_INCL_DIRS))

.PHONY: all

all: c tf py

.PHONY: c

c: $(C_BUILD_DIR)/example
	ln -sf $(C_BUILD_DIR)/example .

$(C_BUILD_DIR)/example: $(C_BUILD_DIR)/example.o $(C_OBJ_PATHS)
	$(CC) -o $@ $^ $(GEN_FLAGS) $(LIB_FLAGS)

$(C_BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(C_INCL_PATHS)
	mkdir -p $(C_BUILD_DIR)
	$(CC) -c -o $@ $< $(GEN_FLAGS) $(C_ALL_INCL_FLAGS)

.PHONY: tf

tf: $(TF_BUILD_DIR)/_dci_tf.so
	ln -sf $(TF_BUILD_DIR)/_dci_tf.so .

$(TF_BUILD_DIR)/%_tf.so: $(SRC_DIR)/tf%.cc $(TF_OBJ_PATHS)
	$(CPPC) -shared -o $@ $^ -fPIC $(LIB_FLAGS) $(TF_ALL_INCL_FLAGS) $(TF_GEN_FLAGS) $(TF_LIB_FLAGS)

$(TF_BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(TF_INCL_PATHS)
	mkdir -p $(TF_BUILD_DIR)
	$(CC) -c -o $@ $< -fPIC $(GEN_FLAGS) $(TF_ALL_INCL_FLAGS)

.PHONY: py

py: $(PY_BUILD_DIR)/_dci.so
	ln -sf $(PY_BUILD_DIR)/_dci.so .
	ln -sf $(SRC_DIR)/dci.py .

$(PY_BUILD_DIR)/%.so: $(PY_BUILD_DIR)/py%.o $(PY_OBJ_PATHS)
	$(CC) -shared -o $@ $^ -fPIC $(GEN_FLAGS) $(LIB_FLAGS)

$(PY_BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(PY_INCL_PATHS)
	mkdir -p $(PY_BUILD_DIR)
	$(CC) -c -o $@ $< -fPIC $(GEN_FLAGS) $(PY_ALL_INCL_FLAGS)

.PHONY: clean

clean: clean-c clean-tf clean-py
	rm -rf $(BUILD_DIR)

.PHONY: clean-c

clean-c:
	rm -rf $(C_BUILD_DIR) example

.PHONY: clean-tf

clean-tf:
	rm -rf $(TF_BUILD_DIR) _dci_tf.so

.PHONY: clean-py

clean-py:
	rm -rf $(PY_BUILD_DIR) *.pyc dci.py _dci.so
