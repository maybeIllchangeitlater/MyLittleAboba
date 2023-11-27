#ifndef MULTILAYERABOBATRON_MODEL_MATRIXMLP_MLAYER_H_
#define MULTILAYERABOBATRON_MODEL_MATRIXMLP_MLAYER_H_
#include "../../Utility/Matrix.h"
namespace s21 {
struct MLayer {
  MLayer() = default;
  MLayer(Matrix &&weights, Matrix &&biases)
      : weights_(std::move(weights)), biases_(std::move(biases)) {}
  MLayer(const MLayer &) = default;
  MLayer(MLayer &&) = default;
  MLayer &operator=(const MLayer &) = default;
  MLayer &operator=(MLayer &&) = default;

  Matrix weights_;
  Matrix biases_;
  Matrix activated_outputs_;
  Matrix error_;
  Matrix outputs_;
};
}  // namespace s21
#endif  // MULTILAYERABOBATRON_MODEL_MATRIXMLP_MLAYER_H_
