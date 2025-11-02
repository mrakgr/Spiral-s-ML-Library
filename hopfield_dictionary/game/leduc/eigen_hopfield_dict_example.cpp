#include <iostream>
#include <Eigen/Dense>

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


int main() {
    Eigen::Matrix<float,3,4> keys;
    keys << 1, 1, 0, 0,
            1, 0, 1, 0,
            0, 1, 1, 1;
    Eigen::Matrix<float,3,4> values;
    values << 1.0/2.0, 0, 1.0/2.0, 0,
              0, 3.0/4.0, 1.0/4.0, 0,
              2.0/3.0, 0, 1.0/3.0, 0;
    Eigen::Matrix<float,2,4> input;
    input << 1, 1, 0, 0,
             0, 1, 1, 1;
    
    Eigen::Matrix<float,2,4,0,2,4> out;
    hopfield_dictionary_get<float,3,4,4,2>(input, keys, values, out, 0.001f); 
    std::cout << out << "\n\n";
}