#ifndef BLACKBOX_H_
#define BLACKBOX_H_

#include <vector>
#include <cstdlib>
#include <algorithm>
#include "ActivationFunction.h"
#include <iostream>
#define __ABOBA_DEBUG_


namespace s21{
constexpr static const double eta = 0.15;
class Neuron;
using Layer = std::vector<Neuron>;
class Net{
public:

    explicit Net(const std::vector<unsigned>& topology);
    void FeedNet(const std::vector<double> &input);
    void BackProp(unsigned result);

private:
    std::vector<Layer> layers_;
    double error_;
};



struct Neuron{
public:
//    explicit Neuron(unsigned my_index_x, unsigned my_index_y, int rows, int cols);


    void GetOutputLayerGradient(double target){
        error_ = (target - output_value_) * ActivationFunction::ActivateDerivative(output_value_); //* eta;
#ifdef __ABOBA_DEBUG_
    std::cout << error_ << " and weight is : " << output_value_ << std::endl;
#endif
    }
//    void GetHiddenLayerGradient(const Layer& next_layer){error_ = GetSumOfDerivatives(next_layer) * ActivationFunction::ActivateDerivative(output_value_) ;} //* eta;
//    void UpdateWeights(Layer &prev_layer);

    double output_value_;
    unsigned my_index_; //neuron index in layer
    double error_;
    double bias_;
    std::vector<double> weights_;
};
}//namespace s21

#endif //BLACKBOX_H_
