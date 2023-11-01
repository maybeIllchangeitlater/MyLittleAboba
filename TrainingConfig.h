#ifndef MULTILAYERABOBATRON_UTILS_TRAININGCONFIG_H_
#define MULTILAYERABOBATRON_UTILS_TRAININGCONFIG_H_
#include <vector>
namespace s21 {
    struct TrainingConfig {
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
        const char *winner_savepath;
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
}

#endif //MULTILAYERABOBATRON_UTILS_TRAININGCONFIG_H_
