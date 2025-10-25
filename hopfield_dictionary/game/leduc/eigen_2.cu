#include <Eigen/Dense>
#include <iostream>

int main() {
    const double temperature = 0.001;
    Eigen::MatrixXd keys(3, 4);
    keys << 1, 1, 0, 0,
            1, 0, 1, 0,
            0, 1, 1, 1;
    Eigen::MatrixXd values(3, 4);
    values << 1.0/2.0, 0, 1.0/2.0, 0,
              0, 3.0/4.0, 1.0/4.0, 0,
              2.0/3.0, 0, 1.0/3.0, 0;
    Eigen::MatrixXd input(2, 4);
    input << 1, 1, 0, 0,
             0, 1, 1, 1;
    
    Eigen::MatrixXd out;
    out.setZero();
    
    Eigen::MatrixXd & out_ref = out;
    // matmul(((matmul(keys, input.T) - input.sum) / temperature).T, values)
    out_ref = ((keys * input.transpose() - input.rowwise().sum().transpose().replicate(keys.rows(),1)).array() / temperature).exp().matrix().transpose() * values;
    std::cout << out << "\n\n";
}
