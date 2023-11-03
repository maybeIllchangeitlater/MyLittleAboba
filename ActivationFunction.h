#ifndef MULTILAYERABOBATRON_UTILS_ACTIVATION_FUNCTION_H_
#define MULTILAYERABOBATRON_UTILS_ACTIVATION_FUNCTION_H_
#include <cmath>
#include <unordered_map>
#include "MLPmatrix.h"

namespace s21 {
    class ActivationFunction {
    public:
        constexpr static double a_gradient = 0.01;


        ///Pros: outputs values between 0 and 1, making it suitable for problems that require probability-like outputs.\n
        ///Cons: can suffer from vanishing gradient slowing down training (bigger network the worse it is),\n
        ///saturation (close to extremes(0, 1) outputs)
        static double Sigmoid(const double x) {
            return 1.0 / (1.0 + std::exp(-x));
        };

        static double SigmoidDeriv(const double x) {
            double s = Sigmoid(x);
            return s * (1.0 - s);
        };


        ///Rectified Linear Unit\n
        ///Pros: efficient, wide range leading to faster convergence, solves vanishing gradient problem\n
        ///Cons: "dead neurons" which can get stack in always outputting 0s, not differentiable at 0
        static double ReLU(const double x) {
            return std::max(0.0, x);
        }

        static double ReLUDeriv(const double x) {
            return x > 0;
        }


        ///Pros: efficient, wide range leading to faster convergence, solves vanishing gradient problem, addresses dead neurons problem\n
        ///Cons: performance can be problem-dependent, The "leakiness" parameter needs to be tuned, choosing the right value can be challenging.
        static double LeakyReLU(const double x) {
            return std::max(a_gradient * x, x);
        }

        static double LeakyReLuDeriv(const double x) {
            return x > 0 ? 1 : a_gradient;
        }


        ///Exponential Linear Unit\n
        ///Pros: ELU can help with faster convergence and mitigate some of the issues associated with ReLU.\n
        ///Unlike to ReLU, ELU can produce negative outputs.\n
        ///Cons: less efficient performance than ReLU and variances\name
        static double ELU(const double x) {
            return x > 0 ? x : a_gradient * (std::pow(std::exp(1.0), x) - 1);
        }

        static double ELUDeriv(const double x) {
            return x > 0 ? 1 : ELU(x) + a_gradient;
        }


        ///Pros: Tanh is zero-centered, which can help with faster convergence in some cases\n
        ///Cons: can suffer from vanishing gradient slowing down training (bigger network the worse it is), but not as bad as Sigmoid\n
        ///saturation but again, not as bad as Sigmoid
        static double Tanh(const double x) {
            return std::tanh(x);
        }

        static double TanhDeriv(const double x) {
            return 1.0 - tanh(x * x);
        }


        ///converts a vector of raw scores into a probability distribution
        static s21::MLPMatrix Softmax(const s21::MLPMatrix &x) {
            s21::MLPMatrix exp_x = x.Exp();
            return exp_x / (exp_x.Sum());
        }

        const static std::unordered_map<const char *, std::pair<double (*)(double), double (*)(double)>> activations_activation_derivatives;

    };
    const std::unordered_map<const char *, std::pair<double (*)(double), double (*)(double)>>
    ActivationFunction::activations_activation_derivatives =
            {{"sigmoid",   std::make_pair(ActivationFunction::Sigmoid, ActivationFunction::SigmoidDeriv)},
            {"tanh",      std::make_pair(ActivationFunction::Tanh, ActivationFunction::TanhDeriv)},
            {"relu",      std::make_pair(ActivationFunction::ReLU, ActivationFunction::ReLUDeriv)},
            {"leakyrelu", std::make_pair(ActivationFunction::LeakyReLU, ActivationFunction::LeakyReLuDeriv)},
            {"elu",       std::make_pair(ActivationFunction::ELU, ActivationFunction::ELUDeriv)}};

}
#endif //MULTILAYERABOBATRON_UTILS_ACTIVATION_FUNCTION_H_
