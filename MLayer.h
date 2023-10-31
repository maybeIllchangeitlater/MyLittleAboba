#ifndef MULTILAYERABOBATRON_MLAYER_H
#define MULTILAYERABOBATRON_MLAYER_H
#include "s21_matrix_oop.h"
namespace s21 {
    struct MLayer {
        MLayer(S21Matrix &&weights, S21Matrix&& biases) : weights_(std::move(weights)), biases_(std::move(biases)) {}
        MLayer(const MLayer&) = default;
        MLayer(MLayer&&) = default;
        MLayer &operator=(MLayer&&) = default;
        MLayer &operator=(const MLayer&) = default;

        MLayer() = default;

        S21Matrix weights_;
        S21Matrix biases_;
        S21Matrix activated_outputs_;
        S21Matrix error_;
        S21Matrix outputs_;
    };
}
#endif //MULTILAYERABOBATRON_MLAYER_H
