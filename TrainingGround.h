#ifndef MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#define MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#include <thread>
#include "MLP.h"
#include "Dataloader.h"
#include "TrainingConfig.h"
#include <ostream>
namespace s21{
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
        ///how many correct answers did each perceptron get
        std::vector<size_t> correctness_counter;
        ///accuracy over training
        std::vector<std::vector<double>> accuracy;
    private:
        void LoadPerceptrons();
        void CreatePerceptrons();
        void TrainPerceptrons(std::vector<std::thread>& they_learn);
        void TestPerceptrons(std::vector<std::thread>& they_learn);
        size_t FindTheBestOne();
        void SaveTheBestOne();
        TrainingConfig& schedule_;
        std::vector<MLP> abobas_;
        DataLoader &dl_;

    };

}


#endif //MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_