#include "GLayer.h"

namespace s21{

    GLayer::GLayer(size_t size, size_t n_size, std::mt19937& gen){

    layer_.resize(size);
    std::uniform_real_distribution<double> dist(2.0/std::sqrt(size * n_size));

    for (size_t i = 0; i < size; ++i){
    layer_[i].bias = 0.0;
    layer_[i].weight.resize(n_size);
    std::for_each(layer_[i].weight.begin(), layer_[i].weight.end(),
    [&gen, &dist](double& x){ x = dist(gen); });
}
}

    void GLayer::SetInputLayer(const std::vector<double> &in){
        for(size_t n = 0; n < in.size(); ++n){
            layer_[n].output = in[n];
            layer_[n].activated_output = in[n];
        }
    }

    void GLayer::CalcOutputLayerActivatedOutput(){
        std::vector<double> last_layer_outputs_exp;
        for(size_t n = 0; n < layer_.size(); ++n){
            last_layer_outputs_exp.emplace_back(std::exp(layer_[n].output));
        }
        double sum = std::accumulate(last_layer_outputs_exp.begin(), last_layer_outputs_exp.end(), 0.0);

        for(size_t n = 0; n < layer_.size(); ++n){
            layer_[n].activated_output =  last_layer_outputs_exp[n]/sum;
        }
    }

    void GLayer::CalculateError(const GLayer &next_layer, double(*afd)(double)){
        for (size_t n = 0; n < layer_.size(); ++n) {
            layer_[n].error = 0;
            for (size_t next_layer_n = 0; next_layer_n < next_layer.Size(); ++next_layer_n) {
                layer_[n].error += next_layer[next_layer_n].error * next_layer[next_layer_n].weight[n];
            }
            layer_[n].error *= afd(layer_[n].output);
        }
    }

    void GLayer::UpdateWeights(const double lr){
        for(size_t n = 0; n < layer_.size(); ++n){
            for(size_t w = 0; w < layer_[n].weight.size(); ++w) {
                layer_[n].weight[w] -= layer_[n].activated_output * layer_[n].error * lr;
            }
        }
        for(auto& neuron : layer_){
            neuron.bias -= neuron.error * lr;
        }
    }

}