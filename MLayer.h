#ifndef MULTILAYERABOBATRON_MLAYER_H
#define MULTILAYERABOBATRON_MLAYER_H
#include "s21_matrix_oop.h"
namespace s21 {
    struct MLayer {
        MLayer(S21Matrix &&w) : weights_(w), is_output_layer_(false) {}

        MLayer() : is_output_layer_(true) {}

        S21Matrix weights_;
        S21Matrix activated_outputs_;
        S21Matrix error_;
        S21Matrix outputs_;
        const bool is_output_layer_;
    };
}
#endif //MULTILAYERABOBATRON_MLAYER_H
