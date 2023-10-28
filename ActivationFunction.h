#ifndef ACTIVATION_FUNCTION_H_
#define ACTIVATION_FUNCTION_H_
#include <cmath>
#include "s21_matrix_oop.h"


class ActivationFunction{
public:
    static double Sigmoid(const double x){
        return 1.0 / (1.0 + std::exp(-x));
    };
    static double SigmoidDeriv(const double x){
        double s = Sigmoid(x);
        return s * (1.0 - s);
    };
    static s21::S21Matrix Softmax(const s21::S21Matrix& x){
        s21::S21Matrix exp_x = x.Exp();
        return exp_x/(exp_x.Sum());
    }
};

#endif //ACTIVATION_FUNCTION_H_
