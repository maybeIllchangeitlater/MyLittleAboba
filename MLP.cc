#include "MLP.h"
#include <algorithm>

namespace s21{

    MLP::MLP(std::vector<size_t> topology, DataLoader * dl, const char* activation_function_name) : dl_(dl),
    activation_function_name_(activation_function_name), gen_(std::random_device()()) {

        GetActivationFunction();

        for(size_t i = 0; i < topology.size() - 1; ++i){
            //initialize layer weights with small random values
            //using Xavier initialization
            //initial biases are 0s
            layers_.emplace_back(Mx(topology[i], topology[i + 1], gen_,
                                           0.0, (2.0/std::sqrt(topology[i] * topology[i + 1]))),
                                 Mx(1, topology[i + 1]));
        }
        layers_.emplace_back();

    }

    void MLP::FeedForward(const Mx &in) {

        layers_[0].activated_outputs_ = in;
        layers_[0].outputs_ = in;

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

    void MLP::BackPropogation(const Mx &ideal) {

        layers_[layers_.size() - 2].error_ = layers_.back().activated_outputs_ - ideal;
        //dZ = a - Y
        for(std::ptrdiff_t i = layers_.size() - 3; i >=0; --i){
            layers_[i].error_ = layers_[i + 1].error_.MulByTransposed(layers_[i + 1].weights_);
            Mx der = layers_[i + 1].outputs_.Transform(activation_derivative_);
            layers_[i].error_ &= der;
            //dZi = dZi+1 * W.T hadamard product with ActDeriv(Zi) python - dZi = dZi+i.dot(W.T) * ActDeriv(Zi)
        }

        UpdateWeights();

    }

    void MLP::UpdateWeights() {

        for (size_t i = 0; i < layers_.size() - 1; ++i) {
            Mx weight_gradients = layers_[i].activated_outputs_.MulSelfTranspose(layers_[i].error_);
            //dW = a.T * dZ
            weight_gradients *= lr_;
            layers_[i].weights_ -= weight_gradients;

            auto tmp = layers_[i].error_ * lr_;
            layers_[i].biases_ -= tmp;
        }

    }


    void MLP::GradientDescent(double lr, size_t epochs,  size_t batch_size, double lr_reduction, size_t reduction_frequency) {

        lr_ = lr;
        double error = 0.0;
        batch_size = std::min(batch_size, dl_->MaximumTests());
        std::uniform_int_distribution<size_t> dist(0, dl_->MaximumTests() - batch_size);

        for (size_t e = 0; e < epochs; ++e) {
            auto batch = dl_->CreateSample(batch_size, dist(gen_), DataLoader::kTrain, true);

            for (size_t b = 0; b < batch_size; ++b) {
                FeedForward(batch[b].second);
                BackPropogation(batch[b].first);
                error += GetError(batch[b].first);
            }

            average_error_.push_back(error/batch_size);
            error = 0.0;

            if (reduction_frequency && !(e + 1 % reduction_frequency)) {
                lr_ -= lr_reduction;
            }
        }

    }

    size_t MLP::Test() {

        auto & test_set = dl_->TestData();
        correct_test_answers = 0;

        for(const auto& input : test_set)
            correct_test_answers += Debug(input);

        return correct_test_answers;
    }

    size_t MLP::Predict(const s21::Mx &in) {
        FeedForward(in);
        return GetAnswer();
    }

    std::ostream &operator<<(std::ostream &out, const MLP &other){

        out << other.activation_function_name_ << " ";
        out << other.layers_.size() << " ";

        for(const auto & l : other.layers_) {
            out << l.activated_outputs_.Cols() << " "; //save topology
        }

        for(const auto& layer : other.layers_){
            out << layer.weights_;
            out << layer.biases_;
        }

        return out;

    }


    std::istream &operator>>(std::istream &in, MLP &other){

        in >> other.activation_function_name_;
        std::cout << other.activation_function_name_ << std::endl;
        other.GetActivationFunction();
        std::vector<size_t> topology;
        size_t t_size;
        in >>  t_size;
        std::cout << t_size << std::endl;

        for(size_t i = 0; i < t_size; ++i) {
            size_t tmp;
            in >> tmp;
            std::cout << tmp << std::endl;
            topology.push_back(tmp);
        }

        if(topology.front() != other.dl_->Inputs() || topology.back() != other.dl_->Outputs())
            throw std::logic_error("MLP operator>>:"
                                   "Inputs and outputs of preceptron must correspond to ins and outs of "
                                   "dataloader");

        for(size_t i = 0; i < topology.size() - 1; ++i){
            Mx w(topology[i], topology[i + 1]);
            Mx b(1, topology[i + 1]);
            in >> w;
            in >> b;
            other.layers_.emplace_back(std::move(w), std::move(b));
        }

        other.layers_.emplace_back();

        return in;

    }


    void MLP::GetActivationFunction() {

        std::transform(activation_function_name_.begin(), activation_function_name_.end(), activation_function_name_.begin(),
                       [](char c){ return std::tolower(c); }); //to lowercase

        activation_function_name_.erase(std::remove_if(activation_function_name_.begin(), activation_function_name_.end(),
                                                  [](char c){ return (std::isspace(c) || c == '_' || c == '\n'); }),
                                   activation_function_name_.end()); //remove whitespaces _ and newlines

        activation_ = ActivationFunction::activations_activation_derivatives.at(activation_function_name_).first;
        activation_derivative_ = ActivationFunction::activations_activation_derivatives.at(activation_function_name_).second;

    }


    double MLP::GetError(const Mx& ideal){
        return (layers_.back().activated_outputs_ - ideal).Abs().Sum()
               /static_cast<double>(layers_.back().activated_outputs_.Size());
    }

    bool MLP::Debug(const std::pair<Mx, Mx>& in){
        FeedForward(in.second);
        return WasRight(in.first);
    }


    bool MLP::WasRight(const Mx &ideal) {
        size_t i = 0;
        for (; i < ideal.Cols() && !ideal(0, i); ++i){}

        size_t ans = GetAnswer();
        return ans == i;
    }

    size_t MLP::GetAnswer(){
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