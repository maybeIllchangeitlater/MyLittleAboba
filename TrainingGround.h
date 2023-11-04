#ifndef MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#define MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#include <thread>
#include "MLP.h"
#include "Dataloader.h"
#include <ostream>
namespace s21{
        /**
         * @brief Customize training plan for multiple MLPs at once!\n
         * In case of less config inputs than perceptron counter, last parameters, including perceptrons themselves, will be duplicated
         * @param perceptron_counter How many perceptrons to run\n Defaulted to amount of physical cores\nCreating more threads than cores
         * will lower performance
         * @param epochs How many epochs to run each MLP for\n Defaulted to 5
         * @param load Load perceptrons from file?\n Defaulted to no
         * @param path_to_perceptrons Load from where?
         * @param save Save the best perceptron to file?\n Defaulted to yes
         * @param winner_savepath Where to? If unspecified saves to local directory
         * @param topologies For each MLP. Input and output (first and last) values must\n
         * match amount of training set inputs and labels(possible outputs)\n Ignored when loading from file
         * @param activation_functions Available options are: sigmoid, ReLU, leaky ReLU, ELU and Tanh\n Defaulted to sigmoid
         * @param learning_rates Recommended starting value from 0.1 to 0.25\n Defaulted to 0.1
         * @param learning_rate_reductions By how much to reduce learning rate
         * @param learning_rate_reduction_frequencies Once per how many epoch to apply\n Defaulted to 0 (never)
         * @param batch_sizes Defaulted to full training dataset
         */
    struct TrainingConfig {

        constexpr static const size_t kDefaultEpochs = 5;
        constexpr static const char* kDefaultActivator = "sigmoid";
        constexpr static const double kDefaultLR = 0.1;
        constexpr static const double kDefaultLRReductionRate = 0.0;
        constexpr static const size_t kDefaultLRReductionFrequency = 0;
        constexpr static const size_t kDefaultBatchSize = SIZE_T_MAX;

        bool load = false;
        bool save = true;
        size_t perceptron_counter;
        std::string winner_savepath = __FILE__;
        std::vector<const char *> path_to_perceptrons;
        std::vector<const char *> activation_functions;
        std::vector<double> learning_rates;
        std::vector<double> learning_rate_reductions;
        std::vector<size_t> learning_rate_reduction_frequencies;
        std::vector<std::vector<size_t>> topologies;
        std::vector<size_t> epochs;
        std::vector<size_t> batch_sizes;

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
         */
        void Train();
        ///how many correct answers did each perceptron get
        std::vector<size_t> correctness_counter;
        ///accuracy over training for each perceptron. actually, error rate. lower is better
        std::vector<std::vector<double>> accuracy;
    private:
        void LoadPerceptrons();
        void CreatePerceptrons();
        /**
         * @brief set unspecified values to default
         */
        void EnsureConfiguration();
        /**
         * @brief  In case of less config inputs than perceptron counter, duplicate last input(or default) parameters
         */
        void TrainPerceptrons(std::vector<std::thread>& they_learn);
        void TestPerceptrons(std::vector<std::thread>& they_learn);
        size_t FindTheBestOne();
        void SaveAccuracy();
        void SaveTheBestOne();


        TrainingConfig& schedule_;
        DataLoader &dl_;
        std::vector<MLP> abobas_;

    };

}


#endif //MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_