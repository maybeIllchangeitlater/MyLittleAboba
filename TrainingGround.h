#ifndef S21_MLP_TRAININGGROUND_H_
#define S21_MLP_TRAININGGROUND_H_
#include <thread>
#include "MLP.h"
#include "dataloader.h"
#include <ostream>
namespace s21{
    struct TrainingConfig{
        ///amount of MLPs to create
        size_t perceptron_counter;
        ///load or make new\n
        ///loading while path is empty will lead to creating percpetron counter new MLPs
        bool load;
        ///vector of paths to saved perceptrons\n
        ///will create copies of last MLP if you'll try to load less perceptrons than perceptron counter\n
        ///if perceptron counter is less than amount of MLP load files, perceptron counter is ignored
        std::vector<const char *> path_to_perceptrons;
        ///folder where to save the winner
        const char * winner_savepath;
        ///topologies for each perceptron (ignored if they are are loaded from file)
        std::vector<std::vector<size_t>> topologies;
        ///vector of pairs of doubles and pairs of double and unsigned,\n
        ///where first value is learning rate, second is lr reduction\n
        /// and third is how often it should be reduced - 1 each epoch
        std::vector<std::pair<double, std::pair<double, size_t>>> learning_rates;
        ///batch size for each perceptron
        std::vector<size_t> batch_sizes;
        ///iterations for batch
        std::vector<size_t> batch_iterations;
        ///epoch for each MLP
        std::vector<size_t> epochs;
    };
    class TrainingGround{
    public:
        TrainingGround() = delete;
        explicit TrainingGround(TrainingConfig& schelude, DataLoader& d);
        TrainingGround(const TrainingGround&) = delete;
        TrainingGround(TrainingGround&&) = delete;
        TrainingGround operator=(const TrainingGround&) = delete;
        TrainingGround operator=(TrainingGround&&) = delete;
        /**
         * @brief launch MLP training with preloaded config\n
         * the best one is saved
         */
        void Train();
        std::vector<size_t> correctness_counter;
    private:
        void LoadPreceptrons();
        void CreatePreceptrons();
        void TrainPreceptrons(std::vector<std::thread>& they_learn);
        void TestPreceptrons(std::vector<std::thread>& they_learn);
        size_t FindTheBestOne();
        void SaveTheBestOne();
        TrainingConfig& schedule_;
        std::vector<MLP> abobas_;
        DataLoader &dl_;

    };

}


#endif //S21_MLP_TRAININGGROUND_H_