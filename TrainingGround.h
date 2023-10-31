#ifndef S21_MLP_TRAININGGROUND_H_
#define S21_MLP_TRAININGGROUND_H_
#include <thread>
#include "MLP.h"
#include "dataloader.h"
#include <ostream>
namespace s21{
    struct TrainingConfig{
        size_t preceptron_counter; //amount of MLPs to create
        bool load; //load or make new
        size_t copy; //how many copies of loaded up aboba to make. will only work with 1 loaded aboba
        std::vector<const char *> path_to_preceptrons; //path to files with saved weights
        const char * winner_savepath; //path to where to save the best
        std::vector<std::vector<size_t>> topologies; //topologies for each preceptron (ignored if abobas are loaded)
        std::vector<std::pair<double, std::pair<double, size_t>>> learning_rates; //vector of pairs of doubles and pairs of double and unsigned,
        // where first value is learning rate, second is lr reduction and third is how often it should be reduced - 1 each epoch
        std::vector<size_t> batch_sizes; //batch size for each aboba
        std::vector<size_t> batch_iterations;
        std::vector<size_t> epochs; // epochs to train each aboba;
    };
    class TrainingGround{
    public:
        TrainingGround() = delete;
        explicit TrainingGround(TrainingConfig& schelude, DataLoader& d);
        TrainingGround(const TrainingGround&) = delete;
        TrainingGround(TrainingGround&&) = delete;
        TrainingGround operator=(const TrainingGround&) = delete;
        TrainingGround operator=(TrainingGround&&) = delete;

        void Train();
        TrainingConfig& schelude;
        std::vector<MLP> abobas;
        DataLoader &dl;
        std::vector<size_t> correctness_counter;

    };

}


#endif //S21_MLP_TRAININGGROUND_H_