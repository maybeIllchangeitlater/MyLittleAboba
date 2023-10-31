#ifndef MODELLOADER_H_
#define MODELLOADER_H_

#include <vector>
//#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <sstream>
#include "s21_matrix_oop.h"
#include <iostream>
#include <__random/random_device.h>
#include <random>

namespace s21{
class DataLoader{
public:
    using Matrix = S21Matrix;
    enum Mode{
        kTest,
        kTrain
    };
    /**
     *
     * @param inputs amount of neurons in input layer
     * @param outputs amount of labels/neurons in output layer
     */
    DataLoader(size_t inputs, size_t outputs) : in_(inputs), out_(outputs) {}
    /**
     * @brief returns entire train dataset
     */
    const std::vector<std::pair<Matrix, Matrix>>& Data() const noexcept { return data_; }
     /**
      * @brief returns entire test dataset
      */
     const std::vector<std::pair<Matrix, Matrix>>& TestData() const noexcept {return test_data_;}
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
    std::vector<std::pair<Matrix, Matrix>> CreateSample(size_t batch_size = 125, size_t start_from = 0,Mode mode = kTrain, bool shuffle = false);
    /**
     * @brief Get maximum possible amount of tests from dataset
     */
    size_t MaximumTests() const noexcept {return data_.size();}
    size_t Inputs() const noexcept { return in_; }
    size_t Outputs() const noexcept { return out_; }
private:
    size_t in_;
    size_t out_;
    std::vector<std::pair<Matrix, Matrix>> data_;
    std::vector<std::pair<Matrix, Matrix>> test_data_;

};
}
#endif //MODELLOADER_H_
