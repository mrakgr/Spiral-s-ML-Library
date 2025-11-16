#include <arrayfire.h>
#include <iostream>
#include <assert.h>

af::array spherical_kms_dictionary_get(
    const af::array& input,
    const af::array& keys,
    const af::array& values) {
    af::array temp = af::matmul(af::transpose(input), keys);
    
    // Get max indices along dimension 0 (rows)
    af::array max_vals, max_indices;
    af::max(max_vals, max_indices, temp, 1);
    af_print(max_indices);
    
    // Use the indices to lookup values from the values array
    // Each row in max_indices will be used to select a row from values
    return af::lookup(values, max_indices, 1);
}

af::array normalize_l2(const af::array &keys) {
    return keys / af::tile(af::sqrt(af::sum(keys * keys, 0)), keys.dims(0));
}
void normalize_l2_inplace(af::array &keys) {
    keys = keys / af::tile(af::sqrt(af::sum(keys * keys, 0)), keys.dims(0));
}

void neural_gas_spherical_kms_dictionary_key_update(
        const af::array& top_input,
        af::array& keys,
        float lr = 0.1
        ) {
    auto input = normalize_l2(top_input);
    af_print(input);
    auto dim_inner = input.dims(0);
    auto dim_keys = keys.dims(1);
    assert (input.dims(1) == 1);
    af::array scores = af::matmul(af::transpose(input), keys);
    af_print(scores);
    
    // Get max indices along dimension 0 (rows)
    af::array max_vals, max_indices;
    af::max(max_vals, max_indices, scores, 1);

    int dispersal = 5;
    auto neural_gas_factor = af::exp(scores / af::tile(max_vals,1,scores.dims(1)) * dispersal - dispersal);
    af_print(neural_gas_factor);
    
    auto tiled_input = af::tile(input,1,dim_keys);
    af_print(tiled_input);
    auto keys_update = tiled_input - af::tile(af::sum(tiled_input * keys, 0), dim_inner) * keys;
    af_print(keys_update);
    keys = normalize_l2(keys + lr * af::tile(neural_gas_factor,keys.dims(0)) * keys_update);
    af_print(keys);
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
        // Normalizes the inputs by the L2 norm.
        keys = keys / af::tile(af::sqrt(af::sum(keys * keys, 0)), 4);
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
            1, 0.5, 0, 0,
        };
        af::array input = af::array(4, 1, input_data);
        af_print(input);

        neural_gas_spherical_kms_dictionary_key_update(input, keys);

        // // Compute the result
        // af::array result = spherical_kms_dictionary_get(input, keys, values);
        
        // // Print the result
        // af_print(result);

    } catch (af::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}