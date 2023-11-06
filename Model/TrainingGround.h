#ifndef MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#define MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_
#include <thread>
#include "MLPCore.h"
#include "MatrixMLP/MatrixMLP.h"
#include "Dataloader.h"
#include <ostream>

namespace s21{
        /**
         * @brief Customize training plan for multiple MLPs at once!\n
         * In case of less config inputs than perceptron counter, last parameters,
         * including perceptrons themselves, will be duplicated
         * @param perceptron_counter How many perceptrons to run\n Defaulted to amount of physical cores\n
         * Creating more threads than cores will lower performance
         * @param mlp_types Types of MLP. kMatrix for matrix, kGraph for graph.\n
         * Defaulted to kMatrix, ignored when loading from file
         * @param epochs How many epochs to run each MLP for\n Defaulted to 5
         * @param load Load perceptrons from files?\n Defaulted to no
         * @param load_path Load from where?
         * @param save Save the best perceptron to file?\n Defaulted to yes
         * @param log Save log?\n Defaulted to yes
         * @param save_path Where to? If unspecified saves to local directory
         * @param topologies For each MLP. Input and output (first and last) values must\n
         * match amount of training set inputs and labels(possible outputs)\n Ignored when loading from file
         * @param activation_functions Available options are: sigmoid, ReLU, leaky ReLU, ELU and Tanh\n
         * Defaulted to sigmoid, ignored when loading from file
         * @param learning_rates Defaulted to 0.03. Do not leave at default
         * @param learning_rate_reductions By how much to reduce learning rate
         * @param learning_rate_reduction_frequencies Once per how many epoch to apply\n Defaulted to 0 (never)
         * @param batch_sizes Defaulted to full training dataset
         */
    struct TrainingConfig {

        using MLPType = MLPCore::MLPType;
        constexpr static const size_t kDefaultEpochs = 5;
        constexpr static const char* kDefaultActivator = "sigmoid";
        constexpr static const double kDefaultLR = 0.031;
        constexpr static const double kDefaultLRReductionRate = 0.0;
        constexpr static const size_t kDefaultLRReductionFrequency = 0;
        constexpr static const size_t kDefaultBatchSize = SIZE_T_MAX;
        constexpr static const MLPType kDefaultMLPType = MLPType::kMatrix;

        bool load = false;
        bool log = true;
        bool save = true;
        size_t perceptron_counter;
        std::string save_path = __FILE__;
        std::vector<const char *> load_path;
        std::vector<MLPType> mlp_types;
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
        ~TrainingGround();
        /**
         * @brief launch MLP training with preloaded config\n
         */
        void Train();
        ///how many correct answers did each perceptron get after testing
        std::vector<size_t> correctness_counter;
        ///accuracy over training for each perceptron. actually, error rate. lower is better
        std::vector<std::vector<double>> accuracy;
    private:
        void LoadPerceptrons();
        void CreatePerceptrons();
        /**
         * @brief constructs an instance of MLPInterface based on either MatrixMLP or GraphMLP
         * @return allocated and constructed * to model\nImportant Allocates memory DO NOT DISCARD
         */
        template<typename ... Args>
        [[nodiscard]]MLPCore* ConstructModel(TrainingConfig::MLPType, Args&&... args){

            MLPCore * model;
            model = new MatrixMLP(std::forward<Args>(args)...);

//            if(type == TrainingConfig::kMatrix){
//                model = new s21::MatrixMLP(std::forward<Args>(args)...);
//            }else if(type == TrainingConfig::kGraph){
//                model = ::new MatrixMLP(std::forward<Args>(args)...);
//            }

            return model;

        }
        /**
         * @brief set empty configuration values to default
         */
        void EnsureConfiguration();
        /**
         * @brief fill unspecified configuration values with last input(or default) values
         */
        void FillMissingConfigurations();
        void TrainPerceptrons(std::vector<std::thread>& they_learn);
        void TestPerceptrons(std::vector<std::thread>& they_learn);
        size_t FindTheBestOne();
        void FixSaveLocation();
        void SaveTheBestOne();
        void SaveLog();


        TrainingConfig& schedule_;
        DataLoader &dl_;
        std::vector<MLPCore *> abobas_;

    };

}


#endif //MULTILAYERABOBATRON_MODEL_TRAININGGROUND_H_