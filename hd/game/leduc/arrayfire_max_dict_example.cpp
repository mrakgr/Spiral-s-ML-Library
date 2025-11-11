#include <arrayfire.h>
#include <iostream>

af::array max_dictionary_get(
    const af::array& input,
    const af::array& keys,
    const af::array& values) {
    af::array temp = af::matmul(af::transpose(input), keys);
    
    // // Get max indices along dimension 0 (rows)
    af::array max_vals, max_indices;
    af::max(max_vals, max_indices, temp + af::randu(temp.dims()), 1);
    af_print(max_indices(0, af::span));
    
    // // Use the indices to lookup values from the values array
    // // Each row in max_indices will be used to select a row from values
    return af::lookup(values, max_indices, 1);
}

int main() {
    try {
        // Initialize the keys matrix
        float keys_data[] = {
            1, 1, 0, 0,
            1, 0, 1, 0,
            0, 1, 1, 1
        };
        af::array keys = af::array(4, 3, keys_data);
        af_print(keys);
        
        // Initialize the values matrix
        float values_data[] = {
            0.5f, 0.0f, 0.5f, 0.0f,
            0.0f, 0.75f, 0.25f, 0.0f,
            2.0f/3.0f, 0.0f, 1.0f/3.0f, 0.0f
        };
        af::array values = af::array(4, 3, values_data);
        af_print(values);
        
        // Initialize the input matrix
        float input_data[] = {
            1, 1, 0, 0,
            0, 1, 1, 1
        };
        af::array input = af::array(4, 2, input_data);
        af_print(input);

        // Compute the result
        af::array result = max_dictionary_get(input, keys, values);
        
        // Print the result
        af_print(result);

    } catch (af::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}