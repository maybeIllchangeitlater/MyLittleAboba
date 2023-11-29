#include "GLayer.h"

namespace s21 {

GLayer::GLayer(size_t p_size, size_t size, std::mt19937& gen) {
  layer_.resize(size);
  std::uniform_real_distribution<double> dist(0.0,
                                              2.0 / std::sqrt(size * p_size));

  for (size_t i = 0; i < size; ++i) {
    layer_[i].bias = 0.0;
    layer_[i].weight.resize(p_size);
    std::for_each(layer_[i].weight.begin(), layer_[i].weight.end(),
                  [&gen, &dist](double& x) { x = dist(gen); });
  }
}
GLayer::GLayer(size_t p_size, size_t size) {
  layer_.resize(size);
  for (size_t i = 0; i < size; ++i) layer_[i].weight.resize(p_size);
}

void GLayer::SetInputLayer(const std::vector<double>& in) {
  for (size_t n = 0; n < in.size(); ++n) {
    layer_[n].output = in[n];
    layer_[n].activated_output = in[n];
  }
}

void GLayer::CalcOutputLayerActivatedOutput() {
  std::vector<double> last_layer_outputs_exp;
  last_layer_outputs_exp.reserve(layer_.size());
  for (auto& n : layer_) {
    last_layer_outputs_exp.emplace_back(std::exp(n.output));
  }
  double sum = std::accumulate(last_layer_outputs_exp.begin(),
                               last_layer_outputs_exp.end(), 0.0);

  for (size_t n = 0; n < layer_.size(); ++n) {
    layer_[n].activated_output = last_layer_outputs_exp[n] / sum;
  }
}

void GLayer::CalculateError(const GLayer& next_layer, double (*afd)(double)) {
  for (size_t n = 0; n < layer_.size(); ++n) {
    layer_[n].error = 0;
    for (size_t n_l_n = 0; n_l_n < next_layer.Size(); ++n_l_n) {
      layer_[n].error += next_layer[n_l_n].error * next_layer[n_l_n].weight[n];
    }
    layer_[n].error *= afd(layer_[n].output);
  }
}

void GLayer::UpdateWeights(const GLayer& prev_layer, const double lr) {
  for (auto& n : layer_) {
    for (size_t w = 0; w < n.weight.size(); ++w) {
      n.weight[w] -= prev_layer[w].activated_output * n.error * lr;
    }
  }
  for (auto& n : layer_) {
    n.bias -= n.error * lr;
  }
}

}  // namespace s21
