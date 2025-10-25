#pragma once

#include <Eigen/Dense>

void hopfield_dictionary_get(Eigen::MatrixXd & input, Eigen::MatrixXd & keys, Eigen::MatrixXd & values, Eigen::MatrixXd & out, double temperature = 0.001);