#ifndef MULTILAYERABOBATRON_MODEL_MATRIXMLP_MATRIXMLP_H_
#define MULTILAYERABOBATRON_MODEL_MATRIXMLP_MATRIXMLP_H_

#include "../../Utility/ActivationFunction.h"
#include "MLayer.h"
#include "../TrainingGround.h"
#include "../MLPInteraface.h"
#include "../Dataloader.h"

namespace s21 {
    class MatrixMLP : public MLPInterface{
    public:
        using Base = MLPInterface;
        explicit MatrixMLP(s21::DataLoader * d) : dl_(d){}
        explicit MatrixMLP(std::vector<size_t> topology, s21::DataLoader * dl, const char* activation_function = "sigmoid");
        MatrixMLP(const MatrixMLP& other) = default;
        MatrixMLP(MatrixMLP&& other) noexcept  = default;
        MatrixMLP &operator=(const MatrixMLP& other) = default;
        MatrixMLP &operator=(MatrixMLP&& other) noexcept = default;
        ~MatrixMLP() override = default;
        /**
         * @brief Get that error down
         * @param lr learning rate. defaulted to 0.03
         * @param epochs to go through. defaulted to 5
         * @param batch_size defaulted to entire dataset
         * @param lr_reduction reduce lr by. defaulted to 0.0
         * @param reduction_frequency per how many epochs to reduce lr. defaulted to 0 (never)
         */
        void GradientDescent(double lr, size_t epochs, size_t batch_size,
                             double lr_reduction, size_t reduction_frequency) override;

        /**
         * @brief run the tests
         * @return amount of correct guesses
         */
        size_t Test() override;
        /**
         * @brief parse input and guess a label
         */
        size_t Predict(const std::vector<double>& in) override;
        /**
         * @brief returns topology of mlp instance
         */
        std::vector<size_t> Topology() override;

//        const std::vector<MLayer>& GetLayers(){ return layers_; }

//        friend std::ostream &operator<<(std::ostream &out, const MatrixMLP &other);

//        friend std::istream &operator>>(std::istream &in, MatrixMLP &other);
    protected:
        void Out(std::ostream &out) const override;
        void In(std::istream &in) override;
    private:
        ///preform forward propogation from input layer in\n
        ///Zi+1 = ai * Wi + bi (Z0 && a0 = in), ai = activation(Zi)
        void FeedForward(const std::vector<double>& in);
        ///preform backward propogation of error.\n
        ///dZ = a - Y for last layer
        ///backpropogate as dZi = dZi+1 * Wi+1.T hadamard product with activation_deriv(Zi+1)
        void BackPropogation(const std::vector<double>& ideal);
        ///knowing output gradients for each layer update weights and biases
        ///dWi = ai.T * dZ
        ///Wi -= dWi*lr
        ///bi -= dZi * lr
        void UpdateWeights();
        ///retrieves activation function and its derivative
        void GetActivationFunction();
        ///check if answer is correct after feed forwarding
        bool WasRight(const std::vector<double>& ideal);
        ///get label MatrixMLP thinks the answer is after feed forwarding
        size_t GetAnswer();
        ///Forward and check if answer was correct
        bool Debug(const std::pair<std::vector<double>, std::vector<double>>& input);
        ///how close to ideal (0) answer was
        double GetError(const std::vector<double>& ideal);


//        size_t correct_test_answers;
        double lr_;
        s21::DataLoader* dl_;
        double(*activation_)(double);
        double(*activation_derivative_)(double);
        std::vector<MLayer> layers_;
//        std::vector<double> average_error_;
//        std::string activation_function_name_;
        std::mt19937 gen_;

    };
}

#endif //MULTILAYERABOBATRON_MODEL_MATRIXMLP_MATRIXMLP_H_
