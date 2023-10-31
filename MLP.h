#ifndef MULTILAYERABOBATRON_MLP_H
#define MULTILAYERABOBATRON_MLP_H

//#include "s21_matrix_oop.h"
//#include "vector"
#include "ActivationFunction.h"
#include "MLayer.h"
#include "dataloader.h"
//#include <random>

namespace s21 {
    class MLP {
    public:
        using Mx = S21Matrix;
        using AF = ActivationFunction;
        explicit MLP(s21::DataLoader * d) : dl_(d){}
        explicit MLP(std::vector<size_t> topology, s21::DataLoader * dl);
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
        void GradientDescent(double lr, size_t epochs = 5, size_t iterations = 100, size_t batch_size = 125, double lr_reduction = 0.0, size_t reduction_frequency = 0);
        /**
         * @param in pair of ideal output matrix and input matrix
         * @return false if MLP was wrong, true if he was right
         */
        bool Predict(const std::pair<S21Matrix, S21Matrix>& in);
        /**
         * @brief run the tests
         * @return amount of correct guesses
         */
        size_t Test();
        /**
         * @return how many tests did MLP pass last test run
         */
        size_t CorrectAnswers() const noexcept { return chad_counter_; };

        const std::vector<MLayer>& GetLayers(){ return layers_; }


        friend std::ostream &operator<<(std::ostream &out, const MLP &other);
        friend std::istream &operator>>(std::istream &in, MLP &other);

    private:
        ///check if answer is correct
        bool Debug(const Mx& ideal);
        ///get label MLP thinks the answer is
        size_t GetAnswer();
        ///how close to ideal answer was
        double GetAccuracy(const Mx& ideal);
        ///preform forward propogation from input layer in\n
        ///Zi+1 = ai * Wi + bi (Z0 && a0 = in), ai = activation(Zi)
        void FeedForward(const Mx &in);
        ///preform backward propogation of error.\n
        ///dZ = a - Y for last layer
        ///backpropogate as dZi = dZi+1 * Wi+1.T mul elementwise with activation_deriv(Zi+1)
        void BackPropogation(const Mx &ideal);
        ///knowing output gradients for each layer update weights and biases
        ///dWi = ai.T * dZ
        ///Wi -= dWi*lr
        ///bi -= dZi * lr
        void UpdateWeights();

        std::vector<MLayer> layers_;
        s21::DataLoader* dl_;
        std::mt19937 gen_;
        double lr_;
        double average_error_;
        size_t chad_counter_;
//        double average_error_old_;

    };
}

#endif //MULTILAYERABOBATRON_MLP_H
