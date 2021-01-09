#include <iostream>
#include <malloc.h>
#include <stdlib.h>
#include <mutex>
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

#include "dci.h"
#include "util.h"
#include "Eigen/Core"
#include "Eigen/Dense"
using namespace tensorflow;

REGISTER_OP("DciNnSearch")
    .Attr("dim: int")
    .Attr("num_comp_indices: int = 2")
    .Attr("num_simp_indices: int = 7")
    .Attr("num_levels: int = 2")
    .Input("data: float64")
    .Input("query: float64")
    .Input("num_neighbours: int32")
    .Input("update_db: bool")
    .Input("construction_prop_to_visit: float64")
    .Input("construction_prop_to_retrieve: float64")
    .Input("construction_field_of_view: int32")
    .Input("query_prop_to_visit: float64")
    .Input("query_prop_to_retrieve: float64")
    .Input("query_field_of_view: int32")
    .Output("nearest_neighbour_ids: int32")
    .Output("nearest_neighbour_dists: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Docs available at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));      // Ensure data has two axes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));      // Ensure query has two axes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &input));      // Ensure num_neighbours is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &input));      // Ensure update_db is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &input));      // Ensure construction_prop_to_visit is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &input));      // Ensure construction_prop_to_retrieve is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &input));      // Ensure construction_field_of_view is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &input));      // Ensure query_prop_to_visit is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &input));      // Ensure query_prop_to_retrieve is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &input));      // Ensure query_field_of_view is a scalar
      int dim;
      ::tensorflow::shape_inference::DimensionHandle input_dim;
      c->GetAttr("dim", &dim);
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 1), dim, &input_dim));    // Ensure data is ? x dim
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(1), 1), dim, &input_dim));    // Ensure query is ? x dim
      c->set_output(0, c->Matrix(c->Dim(c->input(1), 0), c->UnknownDim()));     // nearest_neighbour_ids is query.shape[0] x ?
      c->set_output(1, c->Matrix(c->Dim(c->input(1), 0), c->UnknownDim()));     // nearest_neighbour_dists is query.shape[0] x ?
      return Status::OK();
    })
    .Doc(R"doc(
Performs k-nearest neighbour search using Dynamic Continuous Indexing.

data: A matrix of shape (num of data points) x dim containing the database of points to search over. 
query: A matrix of shape (num of queries) x dim containing the queries to the database. 
num_neighbours: Number of nearest neighbours to return. 
update_db: Whether or not to update the database. 
construction_prop_to_visit
construction_prop_to_retrieve
construction_field_of_view
query_prop_to_visit
query_prop_to_retrieve
query_field_of_view
nearest_neighbour_ids: A matrix of shape (num of queries) x (num of neighbours) containing the indices of the nearest neighbours to each query. 
nearest_neighbour_dists: A matrix of shape (num of queries) x (num of neighbours) containing the Euclidean distances of between the nearest neighbours and the queries. 
)doc");

class DCINNSearchOp : public OpKernel {
    private:
        int dim;
        int num_comp_indices;
        int num_simp_indices;
        int num_levels;
        dci dci_db;
        std::mutex dci_db_mutex;
        int dci_db_reader_count;
        std::mutex dci_db_reader_count_mutex;
    public:

    explicit DCINNSearchOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("dim", &dim));
        OP_REQUIRES_OK(context, context->GetAttr("num_comp_indices", &num_comp_indices));
        OP_REQUIRES_OK(context, context->GetAttr("num_simp_indices", &num_simp_indices));
        OP_REQUIRES_OK(context, context->GetAttr("num_levels", &num_levels));
        
        dci_db_reader_count = 0;
        dci_init(&dci_db, dim, num_comp_indices, num_simp_indices);
    }

    void Compute(OpKernelContext* context) override {
        // Tensorflow Tensor objects (docs at https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)
        const Tensor& data_tensor = context->input(0);
        const Tensor& query_tensor = context->input(1);
        const Tensor& num_neighbours_tensor = context->input(2);
        const Tensor& update_db_tensor = context->input(3);
        
        const Tensor& construction_prop_to_visit_tensor = context->input(4);
        const Tensor& construction_prop_to_retrieve_tensor = context->input(5);
        const Tensor& construction_field_of_view_tensor = context->input(6);
        const Tensor& query_prop_to_visit_tensor = context->input(7);
        const Tensor& query_prop_to_retrieve_tensor = context->input(8);
        const Tensor& query_field_of_view_tensor = context->input(9);
        
        int num_neighbours = num_neighbours_tensor.flat<int32>()(0);
        bool update_db = update_db_tensor.flat<bool>()(0);
        double construction_prop_to_visit = construction_prop_to_visit_tensor.flat<double>()(0);
        double construction_prop_to_retrieve = construction_prop_to_retrieve_tensor.flat<double>()(0);
        int construction_field_of_view = construction_field_of_view_tensor.flat<int32>()(0);
        double query_prop_to_visit = query_prop_to_visit_tensor.flat<double>()(0);
        double query_prop_to_retrieve = query_prop_to_retrieve_tensor.flat<double>()(0);
        int query_field_of_view = query_field_of_view_tensor.flat<int32>()(0);
        
        auto data = data_tensor.flat<double>().data();  // Tensor.flat() returns an Eigen::Tensor object (whose docs are available at https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html)
        auto query = query_tensor.flat<double>().data();
        int num_points = data_tensor.shape().dim_size(0);
        
        if (update_db) {
            dci_query_config construction_query_config;
            construction_query_config.blind = false;
            construction_query_config.num_to_visit = -1;
            construction_query_config.num_to_retrieve = -1;
            construction_query_config.prop_to_visit = construction_prop_to_visit;
            construction_query_config.prop_to_retrieve = construction_prop_to_retrieve;
            construction_query_config.field_of_view = construction_field_of_view;
            
            dci_db_mutex.lock();
            dci_reset(&dci_db);
            dci_add(&dci_db, dim, num_points, (double*) data, num_levels, construction_query_config);
            dci_db_mutex.unlock();
            //std::cout << "Updated database" << std::endl;
        }
        int num_queries = query_tensor.shape().dim_size(0);
        
        int** nearest_neighbour_ids = (int **)malloc(sizeof(int *)*num_queries);
        double** nearest_neighbour_dists = (double **)malloc(sizeof(double *)*num_queries);
        //int* num_returned = (int *)malloc(sizeof(int) * num_queries);
        
        dci_query_config query_config;
        query_config.blind = false;
        query_config.num_to_visit = -1;
        query_config.num_to_retrieve = -1;
        query_config.prop_to_visit = query_prop_to_visit;
        query_config.prop_to_retrieve = query_prop_to_retrieve;
        query_config.field_of_view = query_field_of_view;
        
        dci_db_reader_count_mutex.lock();
        dci_db_reader_count++;
        if (dci_db_reader_count == 1) {
            dci_db_mutex.lock();
        }
        dci_db_reader_count_mutex.unlock();
        
        //dci_query(&dci_db, dim, num_queries, query, num_neighbours, query_config, nearest_neighbour_ids, nearest_neighbour_dists, num_returned);
        dci_query(&dci_db, dim, num_queries, query, num_neighbours, query_config, nearest_neighbour_ids, nearest_neighbour_dists, NULL);
        
        dci_db_reader_count_mutex.lock();
        dci_db_reader_count--;
        if (dci_db_reader_count == 0) {
            dci_db_mutex.unlock();
        }
        dci_db_reader_count_mutex.unlock();
        
        /* Output */
        Tensor* nearest_neighbour_ids_tensor = NULL; 
        Tensor* nearest_neighbour_dists_tensor = NULL;
        TensorShape nearest_neighbour_ids_shape;
        TensorShape nearest_neighbour_dists_shape;
        nearest_neighbour_ids_shape.AddDim(num_queries);
        nearest_neighbour_ids_shape.AddDim(num_neighbours);
        nearest_neighbour_dists_shape.AddDim(num_queries);
        nearest_neighbour_dists_shape.AddDim(num_neighbours);
        OP_REQUIRES_OK(context, context->allocate_output(0, nearest_neighbour_ids_shape, &nearest_neighbour_ids_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, nearest_neighbour_dists_shape, &nearest_neighbour_dists_tensor));
        auto nearest_neighbour_ids_flat = nearest_neighbour_ids_tensor->flat<int32>();
        auto nearest_neighbour_dists_flat = nearest_neighbour_dists_tensor->flat<float>();
        
        for (int i = 0; i < num_queries; i++) {
            for (int j=0; j< num_neighbours; j++){
                nearest_neighbour_ids_flat(num_neighbours*i+j) = nearest_neighbour_ids[i][j];
                nearest_neighbour_dists_flat(num_neighbours*i+j) = nearest_neighbour_dists[i][j];
                //std::cout <<num_neighbours*i+j << " " << nearest_neighbour_ids_flat(num_neighbours*i+j)  << " " << nearest_neighbour_ids[i][j] << " " << nearest_neighbour_dists_flat(num_neighbours*i+j) << " " <<  nearest_neighbour_dists[i][j] << std::endl;
            }
        }  
        for (int i=0; i < num_queries; i++){
            free(nearest_neighbour_ids[i]);
            free(nearest_neighbour_dists[i]);
        }
        free(nearest_neighbour_ids);
        free(nearest_neighbour_dists);
        //free(num_returned);  
    }
};
REGISTER_KERNEL_BUILDER(Name("DciNnSearch").Device(DEVICE_CPU), DCINNSearchOp);
