#include <iostream>
#include "Model/TrainingGround.h"

int main() {
    ///some vals that work for over 80% (784, 128, 128, 26)
    ////all for default batches, no cross validation, no nothing
    ///0.03 lr 5 epochs for sigm reduce by 0.03-4 each
    ///.016 for tanh, relu, elu and leaky relu, reduce by 0.003 after for 5
    s21::DataLoader d(784, 26);
    d.FileToData("/Users/monke/Biba/emnist-letters/emnist-letters-train.csv", s21::DataLoader::kTrain);
    d.FileToData("/Users/monke/Biba/emnist-letters/emnist-letters-test.csv", s21::DataLoader::kTest);

    s21::TrainingConfig tc;
    tc.perceptron_counter = 5;
//    tc.save = false;
//    tc.log = false;
    tc.topologies.push_back(std::vector<size_t>{784, 128, 128, 26});

    tc.activation_functions.emplace_back("sigmoid");
    tc.epochs.emplace_back(5);
    tc.learning_rates.emplace_back(0.03);
    tc.learning_rate_reductions.emplace_back(0.003);
    tc.learning_rate_reduction_frequencies.emplace_back(1);

    tc.activation_functions.emplace_back("tanh");
    tc.learning_rates.emplace_back(0.016);
    tc.activation_functions.emplace_back("ELU");
    tc.activation_functions.emplace_back("ReLU");
    tc.activation_functions.emplace_back("leakyReLU");

//    tc.load = true;
//    tc.path_to_perceptrons.emplace_back("/Users/monke/MultilayerAbobatron/elu_784_128_128_26_correctly_passed_11462tests.txt");
    s21::TrainingGround tg(tc, d);
    tg.Train();


//    output;
//    activated;
//    error;
//    bias;
//    std::vector<double> weight; Node


//


    return 0;
}

///todo dataset to map, graphs, runtime to log
///add load from log?
///add time to log name?

