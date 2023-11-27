#include "MLPBuilder.h"

s21::MLPCore *s21::MLPBuilder::AddMLP(MLPCore::MLPType type,
                                      const std::string &topology,
                                      const std::string &activation) {
  ++schedule_.perceptron_counter;
  std::vector<size_t> tmp;
  size_t num;
  std::istringstream ss(topology);
  while (ss >> num) tmp.emplace_back(num);
  if (tmp.size() < 2)
    throw std::logic_error(
        "TrainingGround: AddMLP: "
        "topology must exist and consist of at least 2 layers");
  if (tmp.front() != dl_.Inputs() || tmp.back() != dl_.Outputs())
    throw std::logic_error(
        "TrainingGround: AddMLP:"
        "topology input layer must correspond to dataset X and topology output"
        "layer must correspond to amount of labels in dataset");
  schedule_.activation_functions.emplace_back(activation.c_str());
  schedule_.mlp_types.emplace_back(type);
  schedule_.topologies.emplace_back(tmp);
  return ConstructModel(type, tmp, &dl_, activation.c_str());
}

std::vector<s21::MLPCore *> s21::MLPBuilder::LoadMLP(
    const std::string &filepath) {
  ++schedule_.perceptron_counter;
  schedule_.load_path.clear();
  char buffer[256];
  std::strcpy(buffer, filepath.c_str());
  buffer[filepath.size()] = '\0';
  schedule_.load_path.emplace_back(buffer);
  return LoadPerceptrons();
}

std::vector<s21::MLPCore *> s21::MLPBuilder::LoadPerceptrons() {
  std::vector<s21::MLPCore *> res;
  for (const auto &s : schedule_.load_path) {
    std::fstream file(s, std::ios_base::in);
    if (file) {
      size_t model_t;
      file >> model_t;

      res.emplace_back(
          ConstructModel(static_cast<MLPCore::MLPType>(model_t), &dl_));

      file >> *res.back();

      file.close();
    } else {
      throw std::invalid_argument(
          "TrainingGround Constructor: load:"
          "specified file doesn't exist");
    }
  }

  for (size_t p = schedule_.load_path.size(); p < schedule_.perceptron_counter;
       ++p)
    res.emplace_back(res.back());
  return res;
}

std::vector<s21::MLPCore *> s21::MLPBuilder::CreatePerceptrons(
    size_t MLPCounter) {
  std::vector<s21::MLPCore *> res;
  for (size_t i = MLPCounter; i < schedule_.perceptron_counter; ++i) {
    if (dl_.Inputs() != schedule_.topologies[i].front() ||
        dl_.Outputs() != schedule_.topologies[i].back()) {
      throw std::logic_error(
          "TrainingGround Constructor:"
          "Inputs and outputs of perceptron must correspond to ins and outs of "
          "dataloader");
    }

    res.emplace_back(ConstructModel(schedule_.mlp_types[i],
                                    schedule_.topologies[i], &dl_,
                                    schedule_.activation_functions[i]));
  }
  return res;
}
