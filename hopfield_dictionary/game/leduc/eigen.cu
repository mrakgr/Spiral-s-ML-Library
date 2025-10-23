#include <iostream>
#include <Eigen/Dense>   // Main Eigen header

int main() {
    using namespace Eigen;

    // Define 2x2 matrix
    Matrix2d mat;
    mat << 1, 2,
           3, 4;

    // Define a 2D vector
    Vector2d vec(5, 6);

    // Matrix-vector multiplication
    Vector2d result = mat * vec;

    std::cout << "Matrix:\n" << mat << "\n\n";
    std::cout << "Vector:\n" << vec << "\n\n";
    std::cout << "Result:\n" << result << "\n";
}
