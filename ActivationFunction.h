#ifndef ACTIVATION_FUNCTION_H_
#define ACTIVATION_FUNCTION_H_


class ActivationFunction{
public:
    static double Activate(double x){
        return 1.0/ (1.0 + std::exp(-x)); //or -a * x. a is slope strength
    };
    static double ActivateDerivative(double x){
        double sig = Activate(x);
        return sig * (1.0 - sig);
    };
};

#endif //ACTIVATION_FUNCTION_H_
