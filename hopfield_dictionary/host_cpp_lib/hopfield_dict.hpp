#pragma once

#include <Eigen/Dense>

void hopfield_dictionary_get(Eigen::MatrixXf & input, Eigen::MatrixXf & keys, Eigen::MatrixXf & values, Eigen::MatrixXf & out, float temperature);