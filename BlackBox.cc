#include "BlackBox.h"
#include <fstream>
#include <random>

namespace s21{

Net::Net(const std::vector<unsigned> &topology) //first layer is always 28 x 28, last layer is always 1 x 26

// last layer output values is the ans
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, 0.01);
    for(auto t : topology){
        layers_.emplace_back();
        for(unsigned n = 0; n < t; ++n){
            layers_.back().emplace_back();
            for(unsigned v = 0; layers_.size() != 1 && v < topology[layers_.size() - 2]; ++v){
                layers_.back()[n].weights_.emplace_back(distribution(generator));
            }
        }
    }
    std::cout << "initialized" << std::endl;
}

void Net::FeedNet(const std::vector<double> &input) {

    for (unsigned i = 0; i < input.size(); ++i) {
        layers_[0][i].output_value_ = input[i];
    }

    for(size_t curr_layer = 1; curr_layer < layers_.size(); ++curr_layer){
        for(size_t curr_neuron = 0; curr_neuron < layers_[curr_layer].size(); ++curr_neuron){
            Neuron& neuron = layers_[curr_layer][curr_neuron];
            double weighted_sum = 0; //sum of Wi *Xi + bias
            for(size_t prev_layer_ni = 0; prev_layer_ni < layers_[curr_layer-1].size(); ++prev_layer_ni){
                weighted_sum += layers_[curr_layer - 1][prev_layer_ni].output_value_ * neuron.weights_[prev_layer_ni];
            } //inner product in other terms. can be replaced with W.transposed * x if switch to matrixes ever happens
            neuron.output_value_ = ActivationFunction::Activate(weighted_sum);
        }
    }

}

void Net::BackProp(unsigned result) {

#ifdef __ABOBA_DEBUG_
        std::cout << "Letter is: " << static_cast<char>('a' + result) << std::endl;
#endif

    for (size_t i = 0; i < layers_.back().size(); ++i){

#ifdef __ABOBA_DEBUG_
        std::cout << "For letter " << static_cast<char>('a' + i) << " error is : ";
#endif
        Neuron& neuron = layers_.back()[i];
        double target = (result == i) ? 1.0 : 0.0;
        neuron.error_ = (target - neuron.output_value_) * ActivationFunction::ActivateDerivative(neuron.output_value_);
#ifdef __ABOBA_DEBUG_
        std::cout <<  neuron.error_ << "\twith result of : " << neuron.output_value_ << std::endl;
#endif
    }

    for (size_t layer = layers_.size() - 2; layer > 0; --layer) {
        for (size_t i = 0; i < layers_[layer].size(); ++i) {
            Neuron& neuron = layers_[layer][i];
            double weighted_sum = 0.0;

            for(size_t j = 0; j < layers_[layer + 1].size(); ++j){
//                weighted_sum += neuron.weights_[j] * layers_[layer + 1][j].error_;
                weighted_sum += layers_[layer + 1][j].weights_[i] * layers_[layer + 1][j].error_;
            }

            neuron.error_ = weighted_sum * ActivationFunction::ActivateDerivative(neuron.output_value_);
        }

        }
        UpdateWeights();
    }

    void Net::UpdateWeights(){
        for(size_t layer = 1; layer <layers_.size(); ++layer){
            for(size_t n = 0; n < layers_[layer].size(); ++n){
                Neuron& neuron = layers_[layer][n];
                for(size_t p_layer_n = 0; p_layer_n < layers_[layer - 1].size(); ++p_layer_n){
//                    neuron.weights_[p_layer_n] += neuron.error_ * layers_[layer - 1][p_layer_n].output_value_ * eta;
                    neuron.weights_[p_layer_n] += (neuron.error_ * layers_[layer - 1][p_layer_n].output_value_
                            - lambda * neuron.weights_[p_layer_n]) * eta;
                    //i want to add penalty to loss function to avoid shock data/white noize
                    //gradient * lr
                }
                //neuron.bias_ += eta * neuron.error_;
            }
        }
    }

    char Net::GetResult(const std::vector<double>& input){
        FeedNet(input);
        double max_value = 10;
        size_t ind;
        for(size_t i = 0; i< layers_.back().size(); ++i){
            if(layers_.back()[i].output_value_ < max_value){
                max_value = layers_.back()[i].output_value_;
                ind = i;
            }
        }
        return static_cast<char>('a' + ind);
    }


}
