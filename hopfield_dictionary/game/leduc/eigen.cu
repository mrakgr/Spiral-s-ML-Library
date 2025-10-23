#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd X(3, 4);
    X << 1, 2, 3, 4,
         2, 4, 6, 8,
         1, 3, 5, 7;

    // Step 1: subtract row-wise max for numerical stability
    Eigen::MatrixXd X_stable = X - X.rowwise().maxCoeff().replicate(1, X.cols());
    
    // // Step 2: exponentiate element-wise
    Eigen::MatrixXd exps = X_stable.array().exp();
    
    // // Step 3: normalize by row-wise sum (broadcasted)
    Eigen::MatrixXd softmax = exps.array() / exps.rowwise().sum().replicate(1, X.cols()).array();
    std::cout << "Input:\n" << softmax << "\n\n";

    // std::cout << "Input:\n" << X << "\n\n";
    // std::cout << "Softmax:\n" << softmax << "\n";
}
