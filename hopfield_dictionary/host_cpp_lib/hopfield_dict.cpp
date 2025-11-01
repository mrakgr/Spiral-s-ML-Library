#include "hopfield_dict.hpp"

// The reason why this function is here instead of in a .cu file is for performance.
// The docs state: "It is thus strongly recommended to properly move all costly host computation from your .cu files to regular .cpp files."
// For more info see: https://libeigen.gitlab.io/eigen/docs-nightly/TopicCUDA.html

// matmul(((matmul(keys, input.T) - input.sum) / temperature).T, values)
// Returns a value (into the output matrix) given the input key from the hopfield dictionary.
template <typename t, int rows, int cols_keys, int cols_values, int rows_input>
void hopfield_dictionary_get(
        Eigen::Matrix<t,rows_input,cols_keys> & input, 
        Eigen::Matrix<t,rows,cols_keys> & keys, 
        Eigen::Matrix<t,rows,cols_values> & values, 
        Eigen::Matrix<t,rows_input,cols_values> & out, 
        t temperature) {
    out = ((keys * input.transpose() - input.rowwise().sum().transpose().replicate(keys.rows(),1)).array() / temperature).exp().matrix().transpose() * values;
}

