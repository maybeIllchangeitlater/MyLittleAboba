#ifndef MULTILAYERABOBATRON_MODEL_GRAPHMLP_GRAPHMLP_H_
#define MULTILAYERABOBATRON_MODEL_GRAPHMLP_GRAPHMLP_H_
#include "../MLPCore.h"
#include "GLayer.h"
#include "../Dataloader.h"

namespace s21 {

    class GraphMLP : public MLPCore{
    public:
        explicit GraphMLP(s21::DataLoader * dl) : MLPCore(dl){}
        explicit GraphMLP(std::vector<size_t> topology, s21::DataLoader * dl, const char* activation_function = "sigmoid");
        GraphMLP(const GraphMLP& other) = default;
        GraphMLP(GraphMLP&& other) noexcept  = default;
        GraphMLP &operator=(const GraphMLP& other) = default;
        GraphMLP &operator=(GraphMLP&& other) noexcept = default;
        ~GraphMLP() override = default;


    protected:
        void FeedForward(const std::vector<double> &in) override;
        void BackPropogation(const std::vector<double> &ideal) override;
        void UpdateWeights() override;
        std::vector<size_t> Topology() const noexcept override;
        double GetError(const std::vector<double>& ideal) const override;
        size_t GetAnswer() const override;
        void Out(std::ostream &out) const override;
        void In(std::istream &in) override;
        DataLoader* dl_;
        std::vector<GLayer> layers_;
        std::mt19937 gen_;
    };

} // s21

#endif //MULTILAYERABOBATRON_MODEL_GRAPHMLP_GRAPHMLP_H_
