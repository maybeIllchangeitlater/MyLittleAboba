#ifndef ACTIVATION_FUNCTION_H_
#define ACTIVATION_FUNCTION_H_
#include <cmath>
#include "MLPmatrix.h"


class ActivationFunction{
public:
    static double Sigmoid(const double x){
        return 1.0 / (1.0 + std::exp(-x));
    };
    static double SigmoidDeriv(const double x){
        double s = Sigmoid(x);
        return s * (1.0 - s);
    };
    static s21::MLPMatrix Softmax(const s21::MLPMatrix& x){
        s21::MLPMatrix exp_x = x.Exp();
        return exp_x/(exp_x.Sum());
    }
};

#endif //ACTIVATION_FUNCTION_H_
