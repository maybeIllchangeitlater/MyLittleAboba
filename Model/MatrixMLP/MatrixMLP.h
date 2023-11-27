#ifndef MULTILAYERABOBATRON_MODEL_MATRIXMLP_MATRIXMLP_H_
#define MULTILAYERABOBATRON_MODEL_MATRIXMLP_MATRIXMLP_H_

#include "../../Utility/ActivationFunction.h"
#include "../Dataloader.h"
#include "../MLPCore.h"
#include "MLayer.h"

namespace s21 {
class MatrixMLP : public MLPCore {
 public:
  explicit MatrixMLP(s21::DataLoader* dl) : MLPCore(dl) {}
  explicit MatrixMLP(std::vector<size_t> topology, s21::DataLoader* dl,
                     const char* activation_function = "sigmoid");
  MatrixMLP(const MatrixMLP& other) = default;
  MatrixMLP(MatrixMLP&& other) noexcept = default;
  MatrixMLP& operator=(const MatrixMLP& other) = default;
  MatrixMLP& operator=(MatrixMLP&& other) noexcept = default;
  ~MatrixMLP() override = default;

  /**
   * @brief returns topology of mlp instance
   */
  std::vector<size_t> Topology() const noexcept override;
  /**
   * @brief Get perceptron model type
   */
  MLPType GetType() override { return MLPType::kMatrix; };

 protected:
  void Out(std::ostream& out) const override;
  void In(std::istream& in) override;
  /// preform forward propogation from input layer in\n
  /// Zi+1 = ai * Wi + bi (Z0 && a0 = in), ai = activation(Zi)
  void FeedForward(const std::vector<double>& in) override;
  /// preform backward propogation of error.\n
  /// dZ = a - Y for last layer
  /// backpropogate as dZi = dZi+1 * Wi+1.T hadamard product with
  /// activation_deriv(Zi+1)
  void BackPropogation(const std::vector<double>& ideal) override;
  /// knowing output gradients for each layer update weights and biases
  /// dWi = ai.T * dZ
  /// Wi -= dWi*lr
  /// bi -= dZi * lr
  void UpdateWeights() override;
  /// how close to ideal (0) answer was
  double GetError(const std::vector<double>& ideal) const override;
  /// get label MatrixMLP thinks the answer is after feed forwarding
  size_t GetAnswer() const override;

  std::vector<MLayer> layers_;
  std::mt19937 gen_;
};
}  // namespace s21

#endif  // MULTILAYERABOBATRON_MODEL_MATRIXMLP_MATRIXMLP_H_
