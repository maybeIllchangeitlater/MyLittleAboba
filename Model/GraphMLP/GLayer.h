#ifndef MULTILAYERABOBATRON_MODEL_GRAPHMLP_GRAPHLAYER_H_
#define MULTILAYERABOBATRON_MODEL_GRAPHMLP_GRAPHLAYER_H_

#include <iostream>
#include <random>
#include <vector>
namespace s21 {
class GLayer {
 public:
  struct GNode {
    double output = 0;
    double activated_output = 0;
    double error = 0;
    double bias = 0.0;
    std::vector<double> weight;
  };

  GLayer() = default;
  explicit GLayer(size_t p_size, size_t size, std::mt19937 &gen);
  explicit GLayer(size_t p_size, size_t size);
  explicit GLayer(size_t size) { layer_.resize(size); };
  GLayer(const GLayer &) = default;
  GLayer(GLayer &&) = default;
  GLayer &operator=(const GLayer &) = default;
  GLayer &operator=(GLayer &&) = default;
  ~GLayer() = default;

  size_t Size() const noexcept { return layer_.size(); }

  void SetInputLayer(const std::vector<double> &in);
  /// Softmax
  void CalcOutputLayerActivatedOutput();
  /// RMS
  void CalculateError(const GLayer &next_layer, double (*afd)(double));
  void UpdateWeights(const GLayer &prev_layer, const double lr);

  GNode &operator[](size_t i) { return layer_[i]; }
  const GNode &operator[](size_t i) const { return layer_[i]; }
  /// save
  friend std::ostream &operator<<(std::ostream &out, const GLayer &other) {
    for (const auto &node : other.layer_)
      for (const auto &w : node.weight) out << w << " ";

    for (const auto &node : other.layer_) out << node.bias << " ";

    return out;
  }
  /// load
  friend std::istream &operator>>(std::istream &in, GLayer &other) {
    for (auto &n : other.layer_)
      for (auto &w : n.weight) in >> w;

    for (auto &n : other.layer_) in >> n.bias;

    return in;
  }

 private:
  std::vector<GNode> layer_;
};

}  // namespace s21

#endif  // MULTILAYERABOBATRON_MODEL_GRAPHMLP_GRAPHLAYER_H_
