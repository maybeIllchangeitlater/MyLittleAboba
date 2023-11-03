#ifndef MULTILAYERABOBATRON_MODEL_MODELLOADER_H_
#define MULTILAYERABOBATRON_MODEL_MODELLOADER_H_

#include <vector>
#include <fstream>
#include <sstream>
#include "MLPmatrix.h"
#include <iostream>
#include <__random/random_device.h>
#include <random>

namespace s21{
    using Mx = MLPMatrix;
    class DataLoader{
public:
    enum Mode{
        kTest,
        kTrain
    };
    /**
     *
     * @param inputs amount of neurons in input layer
     * @param outputs amount of labels/neurons in output layer
     */
    DataLoader(size_t inputs, size_t outputs) : in_(inputs), out_(outputs), gen_(std::random_device()()) {}
    /**
     * @brief returns entire train dataset
     */
    const std::vector<std::pair<Mx, Mx>>& Data() const noexcept { return data_; }
     /**
      * @brief returns entire test dataset
      */
     const std::vector<std::pair<Mx, Mx>>& TestData() const noexcept {return test_data_;}
    /**
     * @brief loads dataset(entire dataset)
     * @param filepath path to dataset
     * @param mode is dataset for testing kTest or training kTrain
     * @param shuffle shuffle dataset
     */
    void FileToData(const char * filepath, Mode mode, bool shuffle = false);
    /**
     * @brief takes sample from dataset
     * @param batch_size amount of samples\n
     * @param start_from start from x input in dataset
     * @param mode test kTest or train kTrain dataset
     * @param shuffle sample
     */
    std::vector<std::pair<Mx, Mx>> CreateSample(size_t batch_size = SIZE_T_MAX, size_t start_from = 0,Mode mode = kTrain, bool shuffle = false);
    /**
     * @brief Get maximum possible amount of samples from learning dataset
     */
    size_t MaximumTests() const noexcept {return data_.size();}
    /**
     * @brief Get maximum possible amount of samples from testing dataset
     */
    size_t MaximumTestsTests() const noexcept { return test_data_.size(); }
    /**
     * @brief get amount of inputs in data
     */
    size_t Inputs() const noexcept { return in_; }
    /**
     * @brief get amount of output labels
     */
    size_t Outputs() const noexcept { return out_; }
private:
    size_t in_;
    size_t out_;
    std::vector<std::pair<Mx, Mx>> data_;
    std::vector<std::pair<Mx, Mx>> test_data_;
    std::mt19937 gen_;

};
}
#endif //MULTILAYERABOBATRON_MODEL_MODELLOADER_H_
