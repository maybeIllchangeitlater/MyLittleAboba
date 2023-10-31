#include "MLP.h"
#include <algorithm>

namespace s21{

    MLP::MLP(std::vector<size_t> topology, DataLoader * dl, double lr) : dl_(dl),
                                                                              gen_(std::random_device()()), lr_(lr){
        for(size_t i = 0; i < topology.size() - 1; ++i){
            //initialize layer weights with small random values using Xavier initialization
            //biases are 0s
            layers_.emplace_back(S21Matrix(topology[i], topology[i + 1], gen_, 0.0, (2.0/std::sqrt(topology[i] * topology[i + 1]))),
                                     S21Matrix(1, topology[i + 1]));
        }
        layers_.emplace_back();
    }

    void MLP::FeedForward(const Mx &in) {
        layers_[0].activated_outputs_ = in;
        layers_[0].outputs_ = in;
        //first layer is input as is
        for(size_t i = 0; i < layers_.size() - 1; ++i){
            layers_[i + 1].outputs_ = layers_[i].activated_outputs_ * layers_[i].weights_;
            layers_[i + 1].outputs_ += layers_[i].biases_;
            //Zi+1 = ai * Wi + bi
            if(i < layers_.size() - 2)
                layers_[i + 1].activated_outputs_ = layers_[i + 1].outputs_.ForEach(AF::Sigmoid);
                //a = activ(Z)
            else
                layers_[i + 1].activated_outputs_ = AF::Softmax(layers_[i + 1].outputs_);
        }
    }

    void MLP::BackPropogation(const Mx &ideal) {
        layers_[layers_.size() - 2].error_ = layers_.back().activated_outputs_ - ideal;
        //dZ = a - Y
        for(int i = layers_.size() - 3; i >=0; --i){
            layers_[i].error_ = layers_[i + 1].error_.MulByTransposed(layers_[i + 1].weights_);
            //dZi = dZi+1 * W.T inner product with ActDeriv(Zi) python - dZi = dZi+i.dot(W.T) * ActDeriv(Zi)
            Mx der = layers_[i + 1].outputs_.ForEach(AF::SigmoidDeriv);
            layers_[i].error_ = layers_[i].error_.MulElementwise(der);
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

    double MLP::GetAccuracy(const Mx& ideal){
        return (layers_.back().activated_outputs_ - ideal).Abs().Sum()
        /static_cast<double>(layers_.back().activated_outputs_.Size());
    }

    bool MLP::Predict(const std::pair<S21Matrix, S21Matrix>& in){
//        for(int r = 0; r < 28; ++r) {
//            for (int c = 0; c < 28; ++c){
//                std::cout << in.second(0 , r*28 + c) << " ";
//            }
//            std::cout << std::endl;
//        }
        FeedForward(in.second);
        return Debug(in.first);
    }

    bool MLP::Debug(const Mx &ideal) {
        size_t i = 0;
        for (; i < ideal.Cols(); ++i) {
            if (ideal(0, i)) {
//                std::cout << "Letter is: " << static_cast<char>('a' + i) << "\t";
                break;
            }
        }
        size_t ans = GetAnswer();
//        std::cout << "Preceptron thinks it is: " << static_cast<char>('a' + ans) << "\n Accuracy is: " <<
//        GetAccuracy(ideal) << std::endl;
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

    size_t MLP::Test() {
        auto & test_set = dl_->TestData();
        chad_counter_ = 0;
        for(const auto& input : test_set){
            chad_counter_ += Predict(input);
        }
        return chad_counter_;
    }

    void MLP::GradientDescent(size_t epochs, size_t iterations, size_t batch_size, double lr_reduction, size_t reduction_frequency){
        average_error_ = 0;
//        average_error_old_ = 0;
        batch_size = std::min(batch_size, dl_->MaximumTests());
        std::uniform_int_distribution<size_t> dist(0, dl_->MaximumTests() - batch_size);
        for(size_t e = 0; e < epochs; ++e) {
            auto batch = dl_->CreateSample(batch_size, dist(gen_));
            for (size_t i = 0; i < iterations; ++i) {
                std::shuffle(batch.begin(), batch.end(), gen_);
                for (int b = 0; b < batch_size; ++b) {
                    FeedForward(batch[b].second);
                    BackPropogation(batch[b].first);
                    if (!i) { //check how accuracy changes for when preceptron first time sees the batch
                        average_error_ += GetAccuracy(batch[b].first);
                    }
                }
                if (!i) {
                    average_error_ = average_error_ / static_cast<double>(batch_size);
//                    std::cout << "average error is " << average_error_ << std::endl;
                    average_error_ = 0;
                }
            }
            if (reduction_frequency && !(e + 1 % reduction_frequency)) {
                lr_ -= lr_reduction;
            }
        }
    }
    std::ostream &operator<<(std::ostream &out, const MLP &other){
        for(const auto & l : other.GetLayer())
            out << l.activated_outputs_.Cols() << " "; //save topology
//        out << "\n" << other.CorrectAnswers() << "\n";
        for(const auto& layer : other.GetLayer()){
            out << layer.weights_;
            out << layer.biases_;
        }
        return out;
    }
    std::istream &operator>>(std::istream &in, MLP &other){
        std::vector<size_t> topology;
        for(int i = 0; i < 4; ++i) {
            size_t tmp;
            in >> tmp;
            topology.push_back(tmp);
        }
        for(int i = 0; i < topology.size(); ++i){
            S21Matrix w(topology[i], topology[i + 1]);
            S21Matrix b(1, topology[i + 1]);
            in >> w;
            in >> b;
            other.GetMutableLayer().emplace_back(std::move(w), std::move(b));
        }
        return in;
    }





}
