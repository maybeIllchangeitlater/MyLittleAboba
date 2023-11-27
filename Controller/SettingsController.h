#ifndef MULTILAYERABOBATRON_CONTROLLER_SETTINGS_CONTROLLER_H_
#define MULTILAYERABOBATRON_CONTROLLER_SETTINGS_CONTROLLER_H_

#include "../Model/TrainingGround.h"

namespace s21 {
class SettingsController {
 public:
  /**
   * @brief constructs settings controller based on MLP training ground
   */
  SettingsController(TrainingConfig& sch) : sch_(sch) {}
  /**
   * @brief Get current amount of perceptrons
   */
  size_t PerceptronsAmount() { return sch_.GetPerceptronCount(); }
  /**
   * @brief Where to save perceptrons and logs
   */
  void SetSavePath(const char* filepath) { sch_.SetSavePath(filepath); }
  /**
   * @brief Size of test batch. By default entire dataset
   */
  void SetTestBatchSize(const size_t size) { sch_.SetTestBatchSize(size); }
  /**
   * @brief Size of train batch. By default entire dataset
   */
  void SetTrainBatchSize(const size_t MLPindex, const size_t size) {
    sch_.SetTrainBatchSize(MLPindex, size);
  }
  /**
   * @brief set lr
   */
  void SetLearningRate(const size_t MLPindex, const double lr) {
    sch_.SetLearningRate(MLPindex, lr);
  }
  /**
   * @brief Reduce learning rate by x.
   */
  void SetLearningRateReduction(const size_t MLPindex, const double reduction) {
    sch_.SetLearningRateReduction(MLPindex, reduction);
  }
  /**
   * @brief Reduce learning rate every x epochs.
   */
  void SetLearningRateReductionFrequency(const size_t MLPindex,
                                         const size_t frequency) {
    sch_.SetLearningRateReductionFrequency(MLPindex, frequency);
  }
  /**
   * @brief epochs to train perceptron for
   */
  void SetEpochs(const size_t MLPindex, const size_t epochs) {
    sch_.SetEpochs(MLPindex, epochs);
  }
  /**
   * @brief Save perceptron to file?
   */
  void SetSave(bool state) { sch_.SetSave(state); }
  /**
   * @brief Save log to file?
   */
  void SetSaveLog(bool state) { sch_.SetSaveLog(state); }
  /**
   * @brief Get path to log file
   */
  std::string& GetLogPath() { return sch_.GetLogPath(); }

 private:
  TrainingConfig& sch_;
};
}  // namespace s21

#endif  // MULTILAYERABOBATRON_CONTROLLER_SETTINGS_CONTROLLER_H_
