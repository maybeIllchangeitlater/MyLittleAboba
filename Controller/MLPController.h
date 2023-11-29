#ifndef MULTILAYERABOBATRON_CONTROLLER_MLPCONTROLLER_H_
#define MULTILAYERABOBATRON_CONTROLLER_MLPCONTROLLER_H_

#include "../Model/MLPBuilder.h"
#include "../Model/TrainingGround.h"

namespace s21 {
class MLPController {
  constexpr static const size_t kInputs = 784;
  constexpr static const size_t kOutputs = 26;

 public:
  /**
   * @brief constructs controller based on MLP training ground
   */
  MLPController(TrainingGround& tg, MLPBuilder& build)
      : tg_(tg), build_(build) {
    tg_.GetDL().SetInputs(kInputs);
    tg_.GetDL().SetOutputs(kOutputs);
  }
  /**
   * @brief Add perceptron of type type with topology topology and activation
   * function activation
   */
  void AddAboba(MLPCore::MLPType type, const std::string& topology,
                const std::string& activation) {
    tg_.GetMLPs().emplace_back(build_.AddMLP(type, topology, activation));
  }
  /**
   * @brief Delete MLP by index
   */
  void DeleteAboba(const size_t MLPindex) { tg_.DeleteMLP(MLPindex); }
  /**
   * @brief Load MLP from file
   */
  void LoadAboba(const std::string& path) {
    auto newboys = build_.LoadMLP(path);
    tg_.GetMLPs().emplace_back(newboys.back());
  }
  void Save(const size_t MLPindex) { tg_.Save(MLPindex); }
  /**
   * @brief GetMLPsInfo as vector of strings with index, model type, topology
   * and activation function
   */
  const std::vector<std::string> GetMLPsInfo() const {
    auto perceptrons = tg_.GetMLPs();
    size_t counter = 1;
    std::vector<std::string> info;
    info.reserve(perceptrons.size());
    for (const auto& p : perceptrons) {
      std::string option(
          std::to_string(counter++) + ". Model of " +
          (p->GetType() == MLPCore::kMatrix ? "matrix" : "graph") +
          " type. Topology: ");
      for (const auto& t : p->Topology()) {
        option += std::to_string(t) + " ";
      }
      option += "Activation Function: " + p->ActivationFunctionName();
      info.emplace_back(std::move(option));
    }
    return info;
  }
  /**
   * @brief Predict a label from input
   */
  size_t Predict(QPixmap& pixmap, size_t MLPindex) {
    return tg_.GetAboba(MLPindex)->Predict(tg_.GetDL().PicToData(pixmap));
  }
  /**
   * @brief run tests on MLP by index
   */
  std::vector<double> Test(size_t MLPindex, double batch_percent) {
    auto* current_mlp = tg_.GetAboba(MLPindex);
    std::vector<double> stats;
    current_mlp->Test(tg_.GetDL().MaximumTestSamples() * batch_percent);
    stats.emplace_back(current_mlp->Accuracy());
    stats.emplace_back(StatHelper(current_mlp->Precision()));
    stats.emplace_back(StatHelper(current_mlp->Recall()));
    stats.emplace_back(StatHelper(current_mlp->F1Score()));
    stats.emplace_back(current_mlp->OutputError().back());
    stats.emplace_back(current_mlp->TestRuntime().count());
    return stats;
  };
  /**
   * @brief Test all MLPs
   */
  void TestAll() {
    Launch();
    tg_.Test();
  };
  /**
   * @brief Train all MLPs
   */
  void Train() {
    Launch();
    tg_.Train();
  };

  void LoadTestsData(const char* filepath) {
    tg_.GetDL().FileToData(filepath, DataLoader::kTest);
  }
  void LoadTrainData(const char* filepath) {
    tg_.GetDL().FileToData(filepath, DataLoader::kTrain);
  }

 private:
  void Launch() {
    auto newboys = build_.Init(tg_.GetMLPs().size());
    if (!newboys.empty())
      tg_.GetMLPs().insert(tg_.GetMLPs().end(), newboys.begin(), newboys.end());
    tg_.Start();
  }
  double StatHelper(const std::vector<double>& stat) {
    double sum = 0.0;
    size_t counter = 0;
    for (; counter < stat.size() && stat[counter] == stat[counter]; ++counter) {
      sum += stat[counter];
    }
    return sum / counter;
  }
  TrainingGround& tg_;
  MLPBuilder& build_;
};
}  // namespace s21

#endif  // MULTILAYERABOBATRON_CONTROLLER_MLPCONTROLLER_H_
