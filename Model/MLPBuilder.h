#ifndef MULTILAYERABOBATRON_MODEL_MLPBUILDER_H_
#define MULTILAYERABOBATRON_MODEL_MLPBUILDER_H_

#include "GraphMLP/GraphMLP.h"
#include "MLPCore.h"
#include "MatrixMLP/MatrixMLP.h"
#include "TrainingConfig.h"

namespace s21 {
class MLPBuilder {
 public:
  MLPBuilder(DataLoader& dl, TrainingConfig& schedule)
      : dl_(dl), schedule_(schedule) {}
  /**
   * @brief Construct MLP model of type type with passed args
   */
  template <typename... Args>
  [[nodiscard]] MLPCore* ConstructModel(MLPCore::MLPType type, Args&&... args) {
    MLPCore* model = nullptr;

    if (type == MLPCore::kMatrix) {
      model = new MatrixMLP(std::forward<Args>(args)...);
    } else if (type == MLPCore::kGraph) {
      model = ::new GraphMLP(std::forward<Args>(args)...);
    }

    return model;
  }
  /**
   * @brief returns MLP with specified settings
   */
  MLPCore* AddMLP(MLPCore::MLPType type, const std::string& topology,
                  const std::string& activation);
  /**
   * @brief returns MLP loaded from file
   */
  std::vector<MLPCore*> LoadMLP(const std::string& filepath);
  /**
   * @brief Initialize perceptrons either by creating new ones or loading from
   * file
   */
  std::vector<MLPCore*> Init(size_t MLPCounter) {
    return schedule_.load && !schedule_.load_path.empty()
               ? LoadPerceptrons()
               : CreatePerceptrons(MLPCounter);
  }

 private:
  DataLoader& dl_;
  std::vector<MLPCore*> LoadPerceptrons();
  std::vector<MLPCore*> CreatePerceptrons(size_t MLPCounter);
  TrainingConfig& schedule_;
};
}  // namespace s21

#endif  // MULTILAYERABOBATRON_MODEL_MLPBUILDER_H_
