#ifndef MULTILAYERABOBATRON_MODEL_DATALOADER_H_
#define MULTILAYERABOBATRON_MODEL_DATALOADER_H_

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace s21 {
class DataLoader {
 public:
  enum Mode { kTest, kTrain };
  /**
   *
   * @param inputs amount of neurons in input layer
   * @param outputs amount of labels/neurons in output layer
   */
  DataLoader(size_t inputs, size_t outputs)
      : in_(inputs), out_(outputs), gen_(std::random_device()()) {}
  /**
   * @brief returns entire train dataset
   */
  const std::unordered_map<size_t, std::vector<std::vector<double>>>& Data()
      const noexcept {
    return data_;
  }
  /**
   * @brief returns entire test dataset
   */
  const std::unordered_map<size_t, std::vector<std::vector<double>>>& TestData()
      const noexcept {
    return test_data_;
  }
  /**
   * @brief loads dataset(entire dataset)
   * @param filepath path to dataset
   * @param mode is dataset for testing kTest or training kTrain
   */
  void FileToData(const char* filepath, Mode mode);
  /**
   * @brief takes sample from train dataset
   * @param batch_size size of sample\n
   * @param mode from test or train dataset
   */
  std::vector<std::pair<size_t, std::vector<double>>> CreateSample(
      size_t batch_size = SIZE_T_MAX, Mode mode = kTrain);
  /**
   * @brief Get maximum possible amount of samples from learning dataset
   */
  size_t MaximumTestSamples() const noexcept { return test_samples_; }
  /**
   * @brief Get maximum possible amount of samples from testing dataset
   */
  size_t MaximumTrainSamples() const noexcept { return train_samples_; }
  /**
   * @brief get amount of inputs in data
   */
  size_t Inputs() const noexcept { return in_; }
  /**
   * @brief get amount of output labels
   */
  size_t Outputs() const noexcept { return out_; }

 private:
  size_t in_;
  size_t out_;
    size_t test_samples_;
    size_t train_samples_;
  std::unordered_map<size_t, std::vector<std::vector<double>>> data_;
  std::unordered_map<size_t, std::vector<std::vector<double>>> test_data_;
  std::mutex mutex_;
  std::mt19937 gen_;
};
}  // namespace s21
#endif  // MULTILAYERABOBATRON_MODEL_DATALOADER_H_
