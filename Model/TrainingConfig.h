#ifndef MULTILAYERABOBATRON_MODEL_TRAININGCONFIG_H_
#define MULTILAYERABOBATRON_MODEL_TRAININGCONFIG_H_
#include "MLPCore.h"
namespace s21 {
class TrainingConfig {
 public:
  /**
   * @brief Where to save perceptrons and logs
   */
  void SetSavePath(const char *filepath) { save_path = filepath; };
  /**
   * @brief Size of test batch. By default entire dataset
   */
  void SetTestBatchSize(const size_t size) { test_batch_size = size; }
  /**
   * @brief Size of train batch. By default entire dataset
   */
  void SetTrainBatchSize(const size_t MLPindex, const size_t size) {
    if (MLPindex > perceptron_counter)
      throw std::out_of_range("SetTrainBatchSize: index out of range");
    if (batch_sizes.size() <= MLPindex) batch_sizes.resize(perceptron_counter);
    batch_sizes[MLPindex] = size;
  }
  /**
   * @brief set lr. by default 0.031. extremely important hyperparametr, don't
   * leave at default
   */
  void SetLearningRate(const size_t MLPindex, const double lr) {
    if (MLPindex > perceptron_counter)
      throw std::out_of_range("SetLearningRate: index out of range");
    if (learning_rates.size() <= MLPindex)
      learning_rates.resize(perceptron_counter);
    learning_rates[MLPindex] = lr;
  }
  /**
   * @brief Reduce learning rate by x. By default 0
   */
  void SetLearningRateReduction(const size_t MLPindex, const double reduction) {
    if (MLPindex > perceptron_counter)
      throw std::out_of_range("SetLearningRateReduction: index out of range");
    if (learning_rate_reductions.size() <= MLPindex)
      learning_rate_reductions.resize(perceptron_counter);
    learning_rate_reductions[MLPindex] = reduction;
  }
  /**
   * @brief Reduce learning rate every x epochs. By default 0 (never)
   */
  void SetLearningRateReductionFrequency(const size_t MLPindex,
                                         const size_t frequency) {
    if (MLPindex > perceptron_counter)
      throw std::out_of_range("SetLearningRateReduction: index out of range");
    if (learning_rate_reduction_frequencies.size() <= MLPindex)
      learning_rate_reduction_frequencies.resize(perceptron_counter);
    learning_rate_reduction_frequencies[MLPindex] = frequency;
  }
  /**
   * @brief epochs to train perceptron for. by default 5
   */
  void SetEpochs(const size_t MLPindex, const size_t epo) {
    if (MLPindex > perceptron_counter)
      throw std::out_of_range("SetEpochs: index out of range");
    if (epochs.size() <= MLPindex) epochs.resize(perceptron_counter);
    epochs[MLPindex] = epo;
  }
  /**
   * @brief Save perceptron to file?
   */
  void SetSave(bool state) { save = state; };
  /**
   * @brief Save log to file?
   */
  void SetSaveLog(bool state) { log = state; };
  /**
   * @brief Get current amount of perceptrons
   */
  size_t GetPerceptronCount() { return perceptron_counter; }
  /**
   * @brief Get path to log file
   */
  std::string &GetLogPath() { return save_path; }

 private:
  constexpr static const size_t kDefaultEpochs = 5;
  constexpr static const char *kDefaultActivator = "sigmoid";
  constexpr static const double kDefaultLR = 0.031;
  constexpr static const double kDefaultLRReductionRate = 0.0;
  constexpr static const size_t kDefaultLRReductionFrequency = 0;
  constexpr static const size_t kDefaultBatchSize = SIZE_T_MAX;
  constexpr static const MLPCore::MLPType kDefaultMLPType =
      MLPCore::MLPType::kMatrix;

  bool load = false;
  bool log = true;
  bool save = true;
  size_t perceptron_counter = 0;
  size_t test_batch_size = SIZE_T_MAX;
  std::string save_path = __FILE__;
  std::vector<char *> load_path;
  std::vector<MLPCore::MLPType> mlp_types;
  std::vector<const char *> activation_functions;
  std::vector<double> learning_rates;
  std::vector<double> learning_rate_reductions;
  std::vector<size_t> learning_rate_reduction_frequencies;
  std::vector<std::vector<size_t>> topologies;
  std::vector<size_t> epochs;
  std::vector<size_t> batch_sizes;
  friend class TrainingGround;
  friend class MLPBuilder;
};
}  // namespace s21
#endif  // MULTILAYERABOBATRON_MODEL_TRAININGCONFIG_H_
