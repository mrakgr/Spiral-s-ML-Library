#include <arrayfire.h>
#include <iostream>

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
    af::array temp = af::matmul(af::transpose(input), keys);
    
    // Get max indices along dimension 0 (rows)
    af::array max_vals, max_indices;
    af::max(max_vals, max_indices, temp, 1);

    auto indexed_keys = af::lookup(keys, max_indices, 1);
    af_print(indexed_keys);
    auto indexed_keys_update = input - af::tile(af::sum(input * indexed_keys, 0), dim_inner) * indexed_keys;
    af_print(indexed_keys_update);
    
    // Create one-hot encoding from max_indices using identity matrix lookup
    // This creates a matrix where each column has a 1 at the position of the matched key
    af::array identity = af::identity(dim_keys, dim_inner);
    af_print(identity);
    af::array one_hot = af::lookup(identity, max_indices, 1);
    af_print(one_hot);
    
    // Accumulate inputs for each key: keys_update = input * one_hot^T
    // This sums all inputs that belong to each key
    af::array sum_inputs = af::matmul(indexed_keys_update, af::transpose(one_hot));
    af::array sum_counts = af::max(1, af::matmul(af::constant(1,indexed_keys_update.dims()), af::transpose(one_hot)));
    af_print(sum_inputs);
    af_print(sum_counts);
    
    // Add accumulated inputs to keys
    keys = normalize_l2(keys + lr * sum_inputs / sum_counts);
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
            0.8, 0.5, 0, 0,
            0, 1, 1, 0.5
        };
        af::array input = af::array(4, 3, input_data);
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