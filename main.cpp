#include <iostream>
#include "MLP.h"
#include "TrainingGround.h"
#include "TrainingConfig.h"
int main() {
    s21::DataLoader d(784, 26);

    d.FileToData("/Users/monke/Biba/emnist-letters/emnist-letters-train.csv", s21::DataLoader::kTrain);
    d.FileToData("/Users/monke/Biba/emnist-letters/emnist-letters-test.csv", s21::DataLoader::kTest);
    std::vector<s21::TrainingConfig> night;
    for(size_t i = 0; i < 50; ++i) {
        night.emplace_back();
        night.back().perceptron_counter = 6;
        for(int j = 0; j < 6; ++j) {
            night.back().topologies.push_back({784, 128, 128, 26});
            night.back().batch_sizes.emplace_back(d.MaximumTests());
            night.back().batch_iterations.emplace_back(1);
        }
        night.back().winner_savepath = "/Users/monke/Biba/";
        night.back().load = false;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>uni_lr(0.09, 0.3);
    std::uniform_int_distribution rate_r(0, 2);
    for(size_t i = 0; i < 50; ++i) {
        for(int j = 0; j < 6; ++j) {
            night[i].epochs.emplace_back(std::min((i + 10 % 20), static_cast<size_t>(5)));
            auto lr = uni_lr(gen);
            std::uniform_real_distribution<double>uni_lr_red(lr/38.0, lr/19.0);
            night[i].learning_rates.emplace_back(lr, std::pair(uni_lr_red(gen), rate_r(gen)));
        }
    }
    for(auto & conf : night){
        s21::TrainingGround OnlyTheStrongestWillSurvive(conf, d);
        OnlyTheStrongestWillSurvive.Train();
        for(const auto& w: OnlyTheStrongestWillSurvive.correctness_counter){
            std::cout << w << " ";
        }

        std::cout << "out of " << d.MaximumTestsTests() << std::endl;
        size_t i = 1;
        for(const auto&ae : OnlyTheStrongestWillSurvive.accuracy) {
            std::cout << i++ << "aboba accuracy over time:" << std::endl;
            for(const auto& e: ae){
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }
    }

//    conf.load = true;
//    conf.winner_savepath = "/Users/monke/Biba/4_";
//    conf.path_to_perceptrons.emplace_back("/Users/monke/Biba/3_784_128_128_26_correctly_passed_11456tests.txt");
//    conf.topologies.push_back({784, 128, 128, 26}); //going a bit more complex now
//    conf.topologies.push_back({784, 128, 128, 26}); //going a bit more complex now
//    conf.topologies.push_back({784, 128, 128, 26}); //going a bit more complex now
//    conf.topologies.push_back({784, 128, 128, 26}); //going a bit more complex now
//    conf.topologies.push_back({784, 128, 128, 26}); //going a bit more complex now
//    conf.topologies.push_back({784, 128, 128, 26}); //going a bit more complex now
//    conf.batch_sizes.emplace_back(d.MaximumTests());
//    conf.batch_sizes.emplace_back(d.MaximumTests());
//    conf.batch_sizes.emplace_back(d.MaximumTests());
//    conf.batch_sizes.emplace_back(d.MaximumTests());
//    conf.batch_sizes.emplace_back(d.MaximumTests());
//    conf.batch_sizes.emplace_back(d.MaximumTests());
//    conf.batch_iterations.emplace_back(1);
//    conf.batch_iterations.emplace_back(1);
//    conf.batch_iterations.emplace_back(1);
//    conf.batch_iterations.emplace_back(1);
//    conf.batch_iterations.emplace_back(1);
//    conf.batch_iterations.emplace_back(1);
//    conf.epochs.emplace_back(3);
//    conf.epochs.emplace_back(3);
//    conf.epochs.emplace_back(3);
//    conf.epochs.emplace_back(3);
//    conf.epochs.emplace_back(3);
//    conf.epochs.emplace_back(3);
//    conf.learning_rates.emplace_back(0.02, std::make_pair(0.0, 70));
//    conf.learning_rates.emplace_back(0.012, std::make_pair(0.0, 70));
//    conf.learning_rates.emplace_back(0.01, std::make_pair(0.0, 70));
//    conf.learning_rates.emplace_back(0.023, std::make_pair(0.0, 200));
//    conf.learning_rates.emplace_back(0.017, std::make_pair(0.0, 200));
//    conf.learning_rates.emplace_back(0.015, std::make_pair(0.0, 200));
//    conf.perceptron_counter = 6;

//    s21::TrainingGround OnlyTheStrongestWillSurvive(conf, d);
//    OnlyTheStrongestWillSurvive.Train();


//    s21::MLP aboba(&d);
//    std::fstream loadfile("/Users/monke/Biba/784_150_26_26_corr_6847.txt");
//    loadfile >> aboba;
//    std::vector<size_t> topology({784, 150, 26, 26});
//    s21::MLP aboba(topology, &d, 0.03);
//    aboba.GradientDescent(8);
//    aboba.Test();
//    std::cout << aboba.CorrectAnswers(); //6847
//    std::string path("/Users/monke/Biba/784_150_26_26_corr_");
//    path += std::to_string(aboba.CorrectAnswers()) + ".txt";
//    std::fstream file(path.c_str(), std::ios_base::out);
//    file << aboba;
    ///change dataloader to load both train and test
    ///add multithreading and mutex to dataloader's data
    ///add MLP breeding ground where little MLPs compete for chance of being less retarded
    ///(join, check best performer, save it)
//    d.FileToData(s21::DataLoader::kTest, true);
//    auto test_data = d.Data();
//    std::uniform_int_distribution<size_t> dist(0, d.MaximumTests() - 26);

//    d.FileToData(s21::DataLoader::kTrain, true);
//    std::random_device rd;
//    std::mt19937 gen(rd());

//    s21::MLP biba(std::vector<size_t>({784, 150, 26, 26}), d, 0.03);
//    for(int i = 0; i <8; ++i) {
//        biba.GradientDescent(100, 150);
//        auto batch = d.CreateSample(200, dist(gen));
//        size_t correct = 0;
//        for (const auto &p: batch) {
//            correct += biba.Predict(p);
//        }
//        std::cout << "correctly guessed " << correct << "out of " << batch.size() << " examples" << std::endl;
//    }
    ///test
//    std::cout << "it's testing time" << std::endl;
//    std::random_device rd; // obtain a random number from hardware
//    std::mt19937 gen(rd()); // seed the generator

//    d.FileToData(s21::DataLoader::kTest, true);
//    auto & test = d.Data();
//    size_t correct = 0;
//    for(const auto& p : test){
//        correct += biba.Predict(p);
//    }
//    std::cout << "correctly guessed " << correct << "out of " << d.MaximumTests() <<" examples" << std::endl;


    ///correctly guessed 7382out of 14800 examples
    ///784, 150, 26, 26 topology
    ///0.03 lr
    ///8 batches 100 epochs 150 size

    ///remove 2 more
//    s21::MyLittleAboba(sample, )
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_int_distribution<unsigned> distribution_letters(0, 25);
//    std::uniform_real_distribution<double> distribution_weights(0, 0.01);
//
//    d.FileToData(s21::DataLoader::kTrain);
//    auto sample = d.CreateSample(200);
//
//    ///init from sample
////        Eigen::MatrixXd input(sample[0][0].data(), 784, 1);
//    Eigen::MatrixXd input(784, 1);
//    for(unsigned i = 0; i < 784; ++i){
//        input(i,0) = sample[0][0][i]; ///just an example whats inside initfunc
//        ///vector of doubles will need to be passed in
//    } ///will need to create vector of matrixes like this or prehaps load data into them
//
//
//    ///init parametrs
//        Eigen::MatrixXd W1(26, 784);
//        for(unsigned i = 0; i < W1.rows(); ++i){
//            for(unsigned j = 0; j < W1.cols(); ++j){
//                W1(i, j) = distribution_weights(gen);
//            }
//        }
//        Eigen::MatrixXd b1(26, 1);
//        for(unsigned i = 0; i < b1.rows(); ++i){
//            for(unsigned j = 0; j < b1.cols(); ++j){
//                b1(i, j) = distribution_weights(gen);
//            }
//        }
//        Eigen::MatrixXd W2(26, 26);
//        for(unsigned i = 0; i < W2.rows(); ++i){
//            for(unsigned j = 0; j < W2.cols(); ++j){
//                W2(i, j) = distribution_weights(gen);
//            }
//        }
//        Eigen::MatrixXd b2(26, 1);
//        for(unsigned i = 0; i < b2.rows(); ++i){
//            for(unsigned j = 0; j < b2.cols(); ++j){
//                b2(i, j) = distribution_weights(gen);
//            }
//        }
//
//        ///forward prop
//        Eigen::MatrixXd Z1 = W1*input + b1;
//        Eigen::MatrixXd A1 = 1.0 / (1.0 + (-Z1).array().exp()); ///sigm
//        Eigen::MatrixXd Z2 = W2 * A1 + b2;
//        Eigen::MatrixXd A2 = 1.0 / (1.0 + (-Z2).array().exp());
//
//        ///output vector thing needed
//        Eigen::MatrixXd Y(Eigen::MatrixXd::Zero(26, 1));
//        Y(0, 0) = 1;
//
//        ///backprop
//        unsigned m = Y.size();
//        Eigen::MatrixXd dZ2 = A2 - Y;
//        Eigen::MatrixXd dW2 = (1.0 / m) * dZ2 *A1.transpose();
//        Eigen::MatrixXd db2 = (1.0 / m) * dZ2;
//        Eigen::MatrixXd dZ1 = (W2.transpose() * dZ2).array() * (A1.array() * (1 - A1.array())); ///sigm deriv
//        Eigen::MatrixXd dW1 = (1.0 / m) * dZ1 *input.transpose();
//        Eigen::MatrixXd db1 = (1.0 / m) * dZ1;
//
//        ///update
//        double alpha = 0.5;
//        W1 -= alpha * dW1;
//        b1 -= alpha * db1;
//        W2 -= alpha * dW2;
//        b2 -= alpha * db2;


    return 0;
}

