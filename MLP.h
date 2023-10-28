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

        explicit MLP(const std::vector<size_t>& topology, s21::DataLoader &dl, double lr);
        std::pair<size_t, double> GetPrediction(const Mx& in);
        void GradientDescent(size_t epochs = 1, size_t batch_size = 125, double lr_reduction = 0.0);

    private:
        void Debug(const Mx& ideal);

        void FeedForward(const Mx &in);

        void BackPropogation(const Mx &ideal);

        void UpdateWeights();

        std::vector<MLayer> layers_;
        const std::vector<size_t> &topology_;
        s21::DataLoader& dl_;
        std::mt19937 gen_;
        double lr_;

    };
}

#endif //MULTILAYERABOBATRON_MLP_H
