#include <arrayfire.h>
#include <iostream>

// Test the basic operations from af.spi
int main() {
    try {
        std::cout << "Testing Arrayfire module operations..." << std::endl;
        
        // Test create with rows=3, cols=4
        int rows = 3;
        int cols = 4;
        af::array arr1(cols, rows); // column-major order
        std::cout << "Created array with dims: " << arr1.dims() << std::endl;
        
        // Test array_rows (should return 3)
        int actual_rows = arr1.dims(1);
        std::cout << "array_rows: " << actual_rows << " (expected 3)" << std::endl;
        
        // Test array_cols (should return 4)
        int actual_cols = arr1.dims(0);
        std::cout << "array_cols: " << actual_cols << " (expected 4)" << std::endl;
        
        // Test array_clear
        af::array arr2 = af::constant(5.0f, cols, rows);
        std::cout << "Before clear:" << std::endl;
        af_print(arr2);
        arr2 = af::constant(0, arr2.dims(), arr2.type());
        std::cout << "After clear:" << std::endl;
        af_print(arr2);
        
        // Test array indexing with (af::span, column_index) for column access
        af::array arr3 = af::constant(1.0f, cols, rows);
        std::cout << "\nOriginal array:" << std::endl;
        af_print(arr3);
        
        arr3(af::span, 1) = af::constant(9.0f, cols, 1);
        std::cout << "After setting column 1:" << std::endl;
        af_print(arr3);
        
        // Test the max_dictionary_get pattern
        std::cout << "\nTesting max dictionary pattern..." << std::endl;
        float keys_data[] = {
            1, 1, 0, 0,
            1, 0, 1, 0,
            0, 1, 1, 1
        };
        af::array keys = af::array(4, 3, keys_data);
        
        float values_data[] = {
            0.5f, 0.0f, 0.5f, 0.0f,
            0.0f, 0.75f, 0.25f, 0.0f,
            0.666f, 0.0f, 0.334f, 0.0f
        };
        af::array values = af::array(4, 3, values_data);
        
        float input_data[] = {
            1, 1, 0, 0
        };
        af::array input = af::array(4, 1, input_data);
        
        // Perform the max dictionary lookup
        af::array temp = af::matmul(af::transpose(input), keys);
        std::cout << "Matrix multiplication result:" << std::endl;
        af_print(temp);
        
        af::array max_vals, max_indices;
        af::max(max_vals, max_indices, temp, 1);
        std::cout << "Max indices:" << std::endl;
        af_print(max_indices);
        
        af::array result = af::lookup(values, max_indices, 1);
        std::cout << "Lookup result:" << std::endl;
        af_print(result);
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (af::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
