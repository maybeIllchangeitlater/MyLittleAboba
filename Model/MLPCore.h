#ifndef MULTILAYERABOBATRON_MODEL_MLPCORE_H
#define MULTILAYERABOBATRON_MODEL_MLPCORE_H
#include <chrono>
#include <string>
#include <vector>

#include "../Utility/ActivationFunction.h"
#include "Dataloader.h"
namespace s21 {
class MLPCore {  // abstract
 public:
  enum MLPType { kMatrix, kGraph };
  MLPCore(DataLoader *dl) : dl_(dl) {}
  virtual ~MLPCore() = default;
  /**
   * @brief Get that error down
   * @param lr learning rate. defaulted to 0.03
   * @param epochs to go through. defaulted to 5
   * @param batch_size defaulted to entire dataset
   * @param lr_reduction reduce lr by. defaulted to 0.0
   * @param reduction_frequency per how many epochs to reduce lr. defaulted to 0
   * (never)
   * @param cross_valiadtion_groups. defaulted to 0 (no cross validation)
   */
  void GradientDescent(double lr = 0.03, size_t epochs = 5,
                       size_t batch_size = SIZE_T_MAX,
                       double lr_reduction = 0.0,
                       size_t reduction_frequency = 0);
  /**
   * @brief parse input and guess a label
   */
  size_t Predict(const std::vector<double> &in) {
    FeedForward(in);
    return GetAnswer();
  }
  /**
   * @brief run the batch_size amount of tests and find model's precision,
   * accuracy,\n recall and F1
   */
  void Test(size_t batch_size = SIZE_T_MAX);
  /**
   * @brief returns topology of mlp instance
   */
  virtual std::vector<size_t> Topology() const noexcept = 0;

  const std::string &ActivationFunctionName() const noexcept {
    return activation_function_name_;
  }
  /**
   * @brief correct / total answers
   */
  const double &Accuracy() const noexcept { return accuracy_; }
  /**
   * @brief true positives / total positives per label\n
   * if label wasn't present in test sample will return INF
   */
  const std::vector<double> &Precision() const noexcept { return precision_; }
  /**
   * @brief true positives / all positives (tp / tp + fn) per label\n
   * if label wasn't present in test sample will return INF
   */
  const std::vector<double> &Recall() const noexcept { return recall_; }
  /**
   * @brief 2 * precision * recall/ precision + recall
   */
  const std::vector<double> &F1Score() const noexcept { return f1_score_; }
  /**
   * @brief output layer error per epoch
   */
  const std::vector<double> &OutputError() const noexcept { return error_; }
  /**
   * @brief find out how long did last training session took
   */
  std::chrono::seconds TrainRuntime() const noexcept { return train_runtime_; }
  /**
   * @brief find out how long did last testing session took
   */
  std::chrono::seconds TestRuntime() const noexcept { return test_runtime_; }
  /**
   * @brief save MLP
   */
  friend std::ostream &operator<<(std::ostream &out, const MLPCore &other) {
    other.Out(out);
    return out;
  }
  /**
   * @brief load MLP
   */
  friend std::istream &operator>>(std::istream &in, MLPCore &other) {
    other.In(in);
    return in;
  }
  /**
   * @brief Get perceptron model type
   */
  virtual MLPType GetType() = 0;

 protected:
  /// preform forward propogation from input layer in\n
  virtual void FeedForward(const std::vector<double> &in) = 0;
  /// back propogate the error
  virtual void BackPropogation(const std::vector<double> &ideal) = 0;
  virtual void UpdateWeights() = 0;
  /// how close to ideal (0) answer was
  virtual double GetError(const std::vector<double> &ideal) const = 0;
  /// get label MatrixMLP thinks the answer is after feed forwarding
  virtual size_t GetAnswer() const = 0;
  /// retrieves activation function and its derivative
  void GetActivationFunction() {
    std::transform(activation_function_name_.begin(),
                   activation_function_name_.end(),
                   activation_function_name_.begin(),
                   [](char c) { return std::tolower(c); });  // to lowercase

    activation_function_name_.erase(
        std::remove_if(
            activation_function_name_.begin(), activation_function_name_.end(),
            [](char c) { return (std::isspace(c) || c == '_' || c == '\n'); }),
        activation_function_name_.end());  // remove whitespaces _ and newlines

    activation_ = ActivationFunction::activations_activation_derivatives
                      .at(activation_function_name_)
                      .first;
    activation_derivative_ =
        ActivationFunction::activations_activation_derivatives
            .at(activation_function_name_)
            .second;
  }
  /// for <<
  virtual void Out(std::ostream &out) const = 0;
  /// for >>
  virtual void In(std::istream &in) = 0;

  double lr_;
  double accuracy_;
  double (*activation_)(double);
  double (*activation_derivative_)(double);
  DataLoader *dl_;
  std::chrono::seconds train_runtime_;
  std::chrono::seconds test_runtime_;
  std::string activation_function_name_;
  std::vector<double> error_;
  std::vector<double> precision_;
  std::vector<double> recall_;
  std::vector<double> f1_score_;
};
}  // namespace s21

#endif  // MULTILAYERABOBATRON_MODEL_MLPCORE_H
