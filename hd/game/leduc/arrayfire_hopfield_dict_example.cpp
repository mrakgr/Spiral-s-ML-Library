#include <arrayfire.h>
#include <iostream>

// matmul(((matmul(keys, input.T) - input.sum) / temperature).T, values)
// Returns a value (into the output matrix) given the input key from the hopfield dictionary.
af::array hopfield_dictionary_get(
    const af::array& input,
    const af::array& keys,
    const af::array& values,
    float temperature) {
    af_print(keys);
    af_print(input);
    
    // Calculate input.sum along rows and replicate
    af::array input_sum = af::sum(input, 1);
    af_print(input_sum);
    af::array replicated_sum = af::transpose(af::tile(input_sum, 1, keys.dims(0)));
    af_print(replicated_sum);
    
    // Main computation
    af::array temp = af::matmul(keys, af::transpose(input));
    af_print(temp);
    temp = temp - replicated_sum;
    temp = af::exp(temp / temperature);
    return af::matmul(af::transpose(temp), values);
}

int main() {
    try {
        // Initialize the keys matrix
        float keys_data[] = {
            1, 1, 0, 0,
            1, 0, 1, 0,
            0, 1, 1, 1
        };
        af::array keys = af::transpose(af::array(4, 3, keys_data)); // Create 4x3 and transpose to get 3x4

        // Initialize the values matrix
        float values_data[] = {
            0.5f, 0.0f, 0.5f, 0.0f,
            0.0f, 0.75f, 0.25f, 0.0f,
            2.0f/3.0f, 0.0f, 1.0f/3.0f, 0.0f
        };
        af::array values = af::transpose(af::array(4, 3, values_data)); // Create 4x3 and transpose to get 3x4

        // Initialize the input matrix
        float input_data[] = {
            1, 1, 0, 0,
            0, 1, 1, 1
        };
        af::array input = af::transpose(af::array(4, 2, input_data)); // Create 4x2 and transpose to get 2x4

        // Compute the result
        af::array result = hopfield_dictionary_get(input, keys, values, 0.001f);
        
        // Print the result
        af_print(result);

    } catch (af::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}