#include "ActivationFunction.h"

namespace s21 {
const std::unordered_map<std::string,
                         std::pair<double (*)(double), double (*)(double)>>
    ActivationFunction::activations_activation_derivatives = {
        {"sigmoid", std::make_pair(ActivationFunction::Sigmoid,
                                   ActivationFunction::SigmoidDeriv)},
        {"tanh", std::make_pair(ActivationFunction::Tanh,
                                ActivationFunction::TanhDeriv)},
        {"relu", std::make_pair(ActivationFunction::ReLU,
                                ActivationFunction::ReLUDeriv)},
        {"leakyrelu", std::make_pair(ActivationFunction::LeakyReLU,
                                     ActivationFunction::LeakyReLuDeriv)},
        {"elu", std::make_pair(ActivationFunction::ELU,
                               ActivationFunction::ELUDeriv)}};
}