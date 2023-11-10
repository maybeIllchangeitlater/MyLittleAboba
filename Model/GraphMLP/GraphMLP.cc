#include "GraphMLP.h"
namespace s21 {
    GraphMLP::GraphMLP(std::vector<size_t> topology,
                       s21::DataLoader * dl, const char* activation_function)
                       : MLPCore(dl), gen_(std::random_device()()){
        activation_function_name_ = activation_function;
        GetActivationFunction();
        layers_.emplace_back(topology[0]);
        for(size_t i = 0; i < topology.size() - 1; ++i){
            layers_.emplace_back(topology[i], topology[i + 1], gen_);
        }
    }

    void GraphMLP::FeedForward(const std::vector<double> &in) {

        layers_[0].SetInputLayer(in);

        for(size_t i = 0; i < layers_.size() - 1; ++i) {
            for (size_t n = 0; n < layers_[i + 1].Size(); ++n) {
                GLayer::GNode &neuron = layers_[i + 1][n];
                neuron.output = 0;
                for(size_t p_n = 0; p_n < layers_[i].Size(); ++p_n){
                    neuron.output += neuron.weight[p_n] * layers_[i][p_n].activated_output;
                }
                neuron.output += neuron.bias;
                if (i < layers_.size() - 2)
                    neuron.activated_output = activation_(neuron.output);
            }
        }
        layers_.back().CalcOutputLayerActivatedOutput();

            //Zi+1 = ai * Wi + bi
            //a = activ(Z)
        }



    void GraphMLP::BackPropogation(const std::vector<double> &ideal) {

        for(size_t n = 0; n < layers_.back().Size(); ++n){
            layers_.back()[n].error = layers_.back()[n].activated_output - ideal[n];
        }
        //dZ = a - Y
        for(std::ptrdiff_t i = layers_.size() - 2; i > 0; --i){
            layers_[i].CalculateError(layers_[i + 1], activation_derivative_);
            //dZi = dZi+1 * W.T hadamard product with ActDeriv(Zi) python - dZi = dZi+i.dot(W.T) * ActDeriv(Zi)
        }

        UpdateWeights();

    }

    void GraphMLP::UpdateWeights() {
        for (size_t i = 1; i < layers_.size(); ++i)
            layers_[i].UpdateWeights(layers_[i-1],lr_);
    }

    std::vector<size_t> GraphMLP::Topology() const noexcept{
        std::vector<size_t> res;
        res.reserve(layers_.size());
        for(const auto& v: layers_)
            res.emplace_back(v.Size());
        return res;
    }


    double GraphMLP::GetError(const std::vector<double>& ideal) const{
        double sum = 0.0;
        for(size_t n = 0; n < layers_.back().Size(); ++n){
            sum += std::pow((layers_.back()[n].activated_output - ideal[n]), 2);
        }
        return std::sqrt(sum);
    }


    size_t GraphMLP::GetAnswer() const{
        double max = 0.0;
        size_t max_i = 0;
        for (size_t n = 0; n < layers_.back().Size(); ++n) {
            if(max < layers_.back()[n].activated_output){
                max = layers_.back()[n].activated_output;
                max_i = n;
            }
        }
        return max_i;
    }

    void GraphMLP::Out(std::ostream &out) const{

        out << kGraph << " ";

        out << activation_function_name_ << " ";
        out << layers_.size() << " ";

        for(const auto & l : layers_) {
            out << l.Size() << " ";
        }

        for(const auto& layer : layers_){
            out << layer;
        }

    }

    void GraphMLP::In(std::istream &in){

        in >> activation_function_name_;
        GetActivationFunction();

        std::vector<size_t> topology;
        size_t t_size;
        in >>  t_size;

        for(size_t i = 0; i < t_size; ++i) {
            size_t tmp;
            in >> tmp;
            topology.push_back(tmp);
        }

        if(topology.front() != dl_->Inputs() || topology.back() != dl_->Outputs())
            throw std::logic_error("GraphMLP operator>>:"
                                   "Inputs and outputs of preceptron must correspond to ins and outs of "
                                   "dataloader");

        for(size_t i = 0; i < topology.size() - 1; ++i){
            in >> layers_[i];
        }

        layers_.emplace_back();
    }


} // s21