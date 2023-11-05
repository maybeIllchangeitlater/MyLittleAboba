#include <iostream>
#include "Model/TrainingGround.h"

int main() {
    ///some vals that work (784, 128, 128, 26)
    ////all for default batches, no cross validation, can be improved with few epochs with heavily reduced lr after
    ///0.03 lr 5 epochs for sigm,
    ///0.03 for relu 2-3 epochs,
    ///0.015 lr 3 epochs leaky relu and elu
    ///start with 0.015 and reduce it each epoch by 0.004 for tanh
    s21::DataLoader d(784, 26);
    d.FileToData("/Users/monke/Biba/emnist-letters/emnist-letters-train.csv", s21::DataLoader::kTrain);
    d.FileToData("/Users/monke/Biba/emnist-letters/emnist-letters-test.csv", s21::DataLoader::kTest);
//    s21::MLP m(std::vector<size_t>{784, 128, 128, 26}, &d, "ReLU");
//        s21::MLP m(&d);
//    std::fstream file_log("/Users/monke/MyLittleAboba/saved configs/elu_784_128_128_26_correctly_passed_12538tests.txt", std::ios_base::in);
//    file_log >> m;
//    m.Test();
//    std::cout << m.CorrectAnswers();

//    m.GradientDescent(0.03, 3, SIZE_T_MAX, 0.01, 1);
//    for(const auto& v: m.GetAccuracy())
//        std::cout << v << " ";
    s21::TrainingConfig tc;
    tc.perceptron_counter = 6;
    tc.save = false;
    tc.topologies.push_back(std::vector<size_t>{784, 128, 128, 26});
    tc.epochs.emplace_back(3);
    tc.activation_functions.emplace_back("ReLU");
    tc.learning_rates.emplace_back(0.015);
    tc.learning_rates.emplace_back(0.03);
    tc.learning_rates.emplace_back(0.02);
    tc.learning_rates.emplace_back(0.025);
    tc.learning_rates.emplace_back(0.01);
    tc.learning_rates.emplace_back(0.035);

//    tc.learning_rate_reductions.emplace_back(0.003);
//    tc.learning_rate_reductions.emplace_back(0.002);
//    tc.learning_rate_reductions.emplace_back(0.001);
//    tc.learning_rate_reductions.emplace_back(0.004);
//    tc.learning_rate_reductions.emplace_back(0.0033);
//    tc.learning_rate_reductions.emplace_back(0.0022);
//    tc.learning_rate_reduction_frequencies.emplace_back(1);
//    tc.learning_rate_reduction_frequencies.emplace_back(1);
//    tc.learning_rate_reduction_frequencies.emplace_back(1);
//    tc.learning_rate_reduction_frequencies.emplace_back(1);
//    tc.learning_rate_reduction_frequencies.emplace_back(0);

//    tc.load = true;
//    tc.path_to_perceptrons.emplace_back("/Users/monke/MultilayerAbobatron/elu_784_128_128_26_correctly_passed_11462tests.txt");
//    }
    s21::TrainingGround tg(tc, d);
    tg.Train();
    for(const auto& w: tg.correctness_counter){
        std::cout << w << " ";
    }
    std::cout << "out of " << d.MaximumTestsTests() << std::endl;

    size_t i = 1;

    for(const auto&ae : tg.accuracy) {
            std::cout << i++ << "aboba accuracy over time:" << std::endl;
            for(const auto& e: ae){
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }



    return 0;
}

