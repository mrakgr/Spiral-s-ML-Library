#include <hopfield_dict.hpp>
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
    
    Eigen::MatrixXd out(2,4);
    hopfield_dictionary_get(input, keys, values, out);
    std::cout << out << "\n\n";
}
