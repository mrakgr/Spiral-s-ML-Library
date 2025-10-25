#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd X(3, 4);
    X << 1, 2, 3, 4,
         2, 4, 6, 8,
         1, 3, 5, 7;
    std::cout << "Input:\n" << X << "\n\n";
    
    Eigen::MatrixXd X_stable = X.colwise() - X.rowwise().maxCoeff();
    std::cout << "X_stable:\n" << X_stable << "\n\n";
    
    // // Step 2: exponentiate element-wise
    // Eigen::MatrixXd exps = X_stable.array().exp();
    // std::cout << "exps:\n" << exps << "\n\n";

    // // Step 3: normalize by row-wise sum (broadcasted)
    // Eigen::MatrixXd softmax = exps.array().colwise() / exps.array().rowwise().sum();
    // std::cout << "Softmax:\n" << softmax << "\n";
}
