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
        explicit MLP(std::vector<size_t> topology, s21::DataLoader * dl, double lr);
        MLP(const MLP& other) = default;
        MLP(MLP&& other) noexcept  = default;
        MLP &operator=(const MLP& other) = default;
        MLP &operator=(MLP&& other) noexcept = default;

        void GradientDescent(size_t epochs = 5, size_t iterations = 100, size_t batch_size = 125, double lr_reduction = 0.0, size_t reduction_frequency = 0);

        bool Predict(const std::pair<S21Matrix, S21Matrix>& in);

        void AdjustLr(double reduction) { lr_ -=reduction;}

        size_t Test();

        size_t CorrectAnswers() const noexcept { return chad_counter_; };

        const std::vector<MLayer>& GetLayer() const noexcept { return layers_; }
        std::vector<MLayer>& GetMutableLayer() noexcept { return layers_; }


        friend std::ostream &operator<<(std::ostream &out, const MLP &other);
        friend std::istream &operator>>(std::istream &in, MLP &other);

    private:
        bool Debug(const Mx& ideal);

        size_t GetAnswer();

        double GetAccuracy(const Mx& ideal);

        void FeedForward(const Mx &in);

        void BackPropogation(const Mx &ideal);

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
