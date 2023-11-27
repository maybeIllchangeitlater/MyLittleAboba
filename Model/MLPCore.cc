#include "MLPCore.h"

namespace s21 {
void MLPCore::GradientDescent(double lr, size_t epochs, size_t batch_size,
                              double lr_reduction, size_t reduction_frequency) {
  auto start = std::chrono::high_resolution_clock::now();
  lr_ = lr;
  double error = 0.0;

  for (size_t e = 0; e < epochs; ++e) {
    auto batch = dl_->CreateSample(batch_size);
    batch_size = batch.size();

    for (size_t b = 0; b < batch_size; ++b) {
      FeedForward(batch[b].second);
      std::vector<double> ideal(dl_->Outputs(), 0);
      ideal[batch[b].first] = 1;
      BackPropogation(ideal);
      error += GetError(ideal);
    }

    error_.push_back(error / batch_size);
    error = 0.0;

    if (reduction_frequency && !((e + 1) % reduction_frequency)) {
      lr_ -= lr_reduction;
    }
  }
  train_runtime_ = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::high_resolution_clock::now() - start);
}

void MLPCore::Test(size_t batch_size) {
  auto start = std::chrono::high_resolution_clock::now();
  accuracy_ = 0;
  auto test_set = dl_->CreateSample(batch_size, DataLoader::kTest);
  accuracy_ = 0;
  precision_.clear();
  recall_.clear();
  f1_score_.clear();
  std::vector<double> true_positives(dl_->Outputs(), 0);
  std::vector<double> false_positives(dl_->Outputs(), 0);
  std::vector<double> false_negatives(dl_->Outputs(), 0);
  size_t predicted_label;
  bool correct;
  double error = 0;

  for (const auto& [label, data] : test_set) {
    predicted_label = Predict(data);
    std::vector<double> ideal(dl_->Outputs(), 0);
    ideal[label] = 1;
    error += GetError(ideal);
    correct = (predicted_label == label);
    if (correct) {
      ++accuracy_;
      ++true_positives[label];
    } else {
      ++false_positives[predicted_label];
      ++false_negatives[label];
    }
  }
  error_.push_back(error / batch_size);

  accuracy_ /= test_set.size();

  for (size_t i = 0; i < dl_->Outputs(); ++i) {
    precision_.emplace_back(true_positives[i] /
                            (true_positives[i] + false_positives[i]));
    recall_.emplace_back(true_positives[i] /
                         (true_positives[i] + false_negatives[i]));
    f1_score_.emplace_back(2 * precision_[i] * recall_[i] / precision_[i] +
                           recall_[i]);
  }
  test_runtime_ = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::high_resolution_clock::now() - start);
}
}  // namespace s21
