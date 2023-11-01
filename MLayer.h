#ifndef MULTILAYERABOBATRON_UTILS_MLAYER_H_
#define MULTILAYERABOBATRON_UTILS_MLAYER_H_
#include "MLPmatrix.h"
namespace s21 {
    using Mx = MLPMatrix;
    struct MLayer {
        MLayer() = default;
        MLayer(Mx &&weights, Mx&& biases) : weights_(std::move(weights)), biases_(std::move(biases)) {}
        MLayer(const MLayer&) = default;
        MLayer(MLayer&&) = default;
        MLayer &operator=(const MLayer&) = default;
        MLayer &operator=(MLayer&&) = default;

        Mx weights_;
        Mx biases_;
        Mx activated_outputs_;
        Mx error_;
        Mx outputs_;
    };
}
#endif //MULTILAYERABOBATRON_UTILS_MLAYER_H_
