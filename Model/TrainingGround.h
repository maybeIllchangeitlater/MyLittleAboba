#ifndef MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#define MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#include <ostream>
#include <thread>

#include "Dataloader.h"
#include "GraphMLP/GraphMLP.h"
#include "MLPBuilder.h"
#include "MLPCore.h"
#include "MatrixMLP/MatrixMLP.h"
#include "TrainingConfig.h"

namespace s21 {
class TrainingGround {
 public:
  TrainingGround() = delete;
  explicit TrainingGround(TrainingConfig& schelude, DataLoader& d);
  TrainingGround(const TrainingGround&) = delete;
  TrainingGround(TrainingGround&&) = delete;
  TrainingGround operator=(const TrainingGround&) = delete;
  TrainingGround operator=(TrainingGround&&) = delete;
  ~TrainingGround();
  /**
   * @brief Load or create perceptrons
   */
  void Start();
  /**
   * @brief launch MLP training with preloaded config\n
   */
  void Train();
  /**
   * @brief Test MLPs\n
   */
  void Test();
  /**
   * @brief Save your favorite aboba, even if he isn't the best
   */
  void Save(const size_t MLPindex);
  void DeleteMLP(const size_t MLPindex);

  MLPCore* GetAboba(size_t index) { return abobas_[index]; }
  std::vector<MLPCore*>& GetMLPs() noexcept { return abobas_; }
  DataLoader& GetDL() { return dl_; }

 private:
  /**
   * @brief set empty configuration values to default
   */
  void EnsureConfiguration();
  /**
   * @brief fill unspecified configuration values with last input(or default)
   * values
   */
  void FillMissingConfigurations();
  void TrainPerceptrons();
  size_t FindTheBestOne();
  void FixSaveLocation();
  void SaveLog();

  TrainingConfig& schedule_;
  DataLoader& dl_;
  std::vector<MLPCore*> abobas_;
};

}  // namespace s21

#endif  // MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
