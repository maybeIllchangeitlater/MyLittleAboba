#include "MatrixMLP.h"
#include <algorithm>

namespace s21{

    MatrixMLP::MatrixMLP(std::vector<size_t> topology, DataLoader * dl, const char* activation_function_name)
    : MLPCore(dl),gen_(std::random_device()()) {

        activation_function_name_ = activation_function_name;
        GetActivationFunction();

        for(size_t i = 0; i < topology.size() - 1; ++i){
            //initialize layer weights with small random values
            //using Xavier initialization
            //initial biases are 0s
            layers_.emplace_back(Matrix(topology[i], topology[i + 1], gen_,
                                           0.0, (2.0/std::sqrt(topology[i] * topology[i + 1]))),
                                 Matrix(1, topology[i + 1]));
        }
        layers_.emplace_back();

    }

    void MatrixMLP::FeedForward(const std::vector<double> &in) {

        layers_[0].outputs_ = in;
        layers_[0].activated_outputs_ = in;

        for(size_t i = 0; i < layers_.size() - 1; ++i){
            layers_[i + 1].outputs_ = layers_[i].activated_outputs_ * layers_[i].weights_;
            layers_[i + 1].outputs_ += layers_[i].biases_;
            layers_[i + 1].activated_outputs_ = i < layers_.size() - 2
                                                ? layers_[i + 1].outputs_.Transform(activation_)
                                                :  ActivationFunction::Softmax(layers_[i + 1].outputs_);
            //Zi+1 = ai * Wi + bi
            //a = activ(Z)
        }

    }

    void MatrixMLP::BackPropogation(const std::vector<double> &ideal) {

        layers_[layers_.size() - 2].error_ = layers_.back().activated_outputs_ - ideal;
        //dZ = a - Y
        for(std::ptrdiff_t i = layers_.size() - 3; i >=0; --i){
            layers_[i].error_ = layers_[i + 1].error_.MulByTransposed(layers_[i + 1].weights_);
            Matrix der = layers_[i + 1].outputs_.Transform(activation_derivative_);
            layers_[i].error_ &= der;
            //dZi = dZi+1 * W.T hadamard product with ActDeriv(Zi) python - dZi = dZi+1.dot(W.T) * ActDeriv(Zi)
        }

        UpdateWeights();

    }

    void MatrixMLP::UpdateWeights() {

        for (size_t i = 0; i < layers_.size() - 1; ++i) {
            Matrix weight_gradients = layers_[i].activated_outputs_.MulSelfTranspose(layers_[i].error_);
            //dW = a.T * dZ
            weight_gradients *= lr_;
            layers_[i].weights_ -= weight_gradients;

            auto tmp = layers_[i].error_ * lr_;
            layers_[i].biases_ -= tmp;
        }

    }


    void MatrixMLP::Out(std::ostream &out) const{

        out << kMatrix << " ";

        out << activation_function_name_ << " ";
        out << layers_.size() << " ";

        for(const auto & l : layers_) {
            out << l.activated_outputs_.Cols() << " "; //save topology
        }

        for(const auto& layer : layers_){
            out << layer.weights_;
            out << layer.biases_;
        }

    }

    void MatrixMLP::In(std::istream &in){

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
            throw std::logic_error("MatrixMLP operator>>:"
                                   "Inputs and outputs of preceptron must correspond to ins and outs of "
                                   "dataloader");

        for(size_t i = 0; i < topology.size() - 1; ++i){
            Matrix w(topology[i], topology[i + 1]);
            Matrix b(1, topology[i + 1]);
            in >> w;
            in >> b;
            layers_.emplace_back(std::move(w), std::move(b));
        }

        layers_.emplace_back();
    }

    std::vector<size_t> MatrixMLP::Topology() const noexcept{
        std::vector<size_t> res;
        for(const auto& v: layers_)
            res.emplace_back(v.outputs_.Size());
        return res;
    }


    double MatrixMLP::GetError(const std::vector<double>& ideal) const{
        return std::sqrt((layers_.back().activated_outputs_ - ideal).Pow2().Sum()
               /static_cast<double>(layers_.back().activated_outputs_.Size()));
    }


    size_t MatrixMLP::GetAnswer() const{
        double max = 0.0;
        size_t max_i = 0;
        for (size_t i = 0; i < layers_.back().activated_outputs_.Cols(); ++i) {
            if(max < layers_.back().activated_outputs_(0, i)){
                max = layers_.back().activated_outputs_(0, i);
                max_i = i;
            }
        }
        return max_i;
    }



}