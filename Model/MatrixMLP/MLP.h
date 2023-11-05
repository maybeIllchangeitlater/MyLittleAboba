#ifndef MULTILAYERABOBATRON_MODEL_MATRIXMLP_MLP_H_
#define MULTILAYERABOBATRON_MODEL_MATRIXMLP_MLP_H_

#include "../../Utility/ActivationFunction.h"
#include "MLayer.h"
#include "../Dataloader.h"

namespace s21 {
    class MLP {
    public:
        explicit MLP(s21::DataLoader * d) : dl_(d){}
        explicit MLP(std::vector<size_t> topology, s21::DataLoader * dl, const char* activation_function = "sigmoid");
        MLP(const MLP& other) = default;
        MLP(MLP&& other) noexcept  = default;
        MLP &operator=(const MLP& other) = default;
        MLP &operator=(MLP&& other) noexcept = default;
        /**
         * @brief Get that error down
         * @param epochs to go through
         * @param iterations spent on each batch
         * @param batch_size
         * @param lr learning_rate
         * @param lr_reduction reduce lr by every
         * @param reduction_frequency epochs
         */
        void GradientDescent(double lr = 0.1, size_t epochs = 5, size_t batch_size = SIZE_T_MAX, double lr_reduction = 0.0, size_t reduction_frequency = 0);

        /**
         * @brief run the tests
         * @return amount of correct guesses
         */
        size_t Test();
        /**
         * @return how many tests did MLP pass last test run
         */
        size_t CorrectAnswers() const noexcept { return correct_test_answers; };
        /**
         * @brief parse input and guess a label
         */
        size_t Predict(const Mx& in);

        const std::string& ActivationFunctionName() { return activation_function_name_; }


        const std::vector<double>& GetAccuracy() const noexcept { return average_error_; }

        const std::vector<MLayer>& GetLayers(){ return layers_; }

        friend std::ostream &operator<<(std::ostream &out, const MLP &other);

        friend std::istream &operator>>(std::istream &in, MLP &other);

    private:
        ///preform forward propogation from input layer in\n
        ///Zi+1 = ai * Wi + bi (Z0 && a0 = in), ai = activation(Zi)
        void FeedForward(const Mx &in);
        ///preform backward propogation of error.\n
        ///dZ = a - Y for last layer
        ///backpropogate as dZi = dZi+1 * Wi+1.T hadamard product with activation_deriv(Zi+1)
        void BackPropogation(const Mx &ideal);
        ///knowing output gradients for each layer update weights and biases
        ///dWi = ai.T * dZ
        ///Wi -= dWi*lr
        ///bi -= dZi * lr
        void UpdateWeights();
        ///retrieves activation function and its derivative
        void GetActivationFunction();
        ///check if answer is correct after feed forwarding
        bool WasRight(const Mx& ideal);
        ///get label MLP thinks the answer is after feed forwarding
        size_t GetAnswer();
        ///Forward and check if answer was correct
        bool Debug(const std::pair<Mx, Mx>& in);
        ///how close to ideal (0) answer was
        double GetError(const Mx& ideal);


        size_t correct_test_answers;
        double lr_;
        s21::DataLoader* dl_;
        double(*activation_)(double);
        double(*activation_derivative_)(double);
        std::vector<MLayer> layers_;
        std::vector<double> average_error_;
        std::string activation_function_name_;
        std::mt19937 gen_;

    };
}

#endif //MULTILAYERABOBATRON_MODEL_MATRIXMLP_MLP_H_
