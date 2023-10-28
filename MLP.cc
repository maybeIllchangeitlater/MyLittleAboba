#include "MLP.h"

namespace s21{

    MLP::MLP(const std::vector<size_t>& topology, DataLoader &dl, double lr) :topology_(topology), dl_(dl),
    gen_(std::random_device()()), lr_(lr){
        if(dl_.Data().empty())
            dl_.FileToData(DataLoader::kTrain);
        for(size_t i = 0; i < topology_.size(); ++i){
            if(i != (topology_.size() - 1))
                //initialize layer weights with small random values using Xavier initialization
                layers_.emplace_back(S21Matrix(topology_[i], topology_[i + 1], gen_, 0.0, (1.0/std::sqrt(topology_[i] * topology_[i + 1]))));
                //2.0/ might be better for sigmoid
            else
                layers_.emplace_back();
        }
    }

    void MLP::FeedForward(const Mx &in) {
        layers_[0].activated_outputs_ = in;
        layers_[0].outputs_ = in;
        //first layer is input as is
        for(size_t i = 0; i < layers_.size() - 1; ++i){
            layers_[i + 1].outputs_ = layers_[i].activated_outputs_ * layers_[i].weights_;
            //Zi+1 = ai * Wi
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
        }
    }

//    std::pair<size_t, double> MLP::GetPrediction(const Mx& in){
//        FeedForward(in);
//        for(size_t i = 0; i < layers_.back().size(); ++i){
//
//        }
//    }
    void MLP::Debug(const Mx &ideal) {
        for (size_t i = 0; i < ideal.GetCols(); ++i) {
            if (ideal(0, i)) {
                std::cout << "Letter is: " << static_cast<char>('a' + i) << "\t";
                break;
            }
        }
        double max = 0.0;
        size_t max_i = 0;
        for (size_t i = 0; i < ideal.GetCols(); ++i) {
            if(max < layers_.back().activated_outputs_(0, i)){
                max = layers_.back().activated_outputs_(0, i);
                max_i = i;
            }
        }
        std::cout << "Preceptron thinks it is: " << static_cast<char>('a' + max_i) << std::endl;
    }

    void MLP::GradientDescent(size_t epochs, size_t batch_size, double lr_reduction){
        batch_size = std::min(batch_size, dl_.MaximumTests());
        std::uniform_int_distribution<size_t> dist(0, dl_.MaximumTests() - batch_size);
        for(size_t e = 0; e < epochs; ++e){
            auto batch = dl_.CreateSample(batch_size, dist(gen_));
            for(int i = 0; i < batch_size; ++i){
                FeedForward(batch[i].second);
                BackPropogation(batch[i].first);
                if(((i + 1 )% 10 == 0)) { Debug(batch[i].first);
                    for(int r = 0; r < 28; ++r) {
                        for (int c = 0; c < 28; ++c){
                            std::cout << batch[i].second(0 , r*28 + c) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            }
            lr_ -= lr_reduction;
        }

    }



///seems promising. add bias, experiment with lr and topology
}
