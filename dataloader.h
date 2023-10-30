#ifndef MODELLOADER_H_
#define MODELLOADER_H_

#include <vector>
//#include "eigen-3.4.0/Eigen/Dense"
#include "s21_matrix_oop.h"
#include <iostream>
#include <__random/random_device.h>
#include <random>

namespace s21{
class DataLoader{
public:
    using Matrix = S21Matrix;
    enum Mode{
      kTrain,
        kTest
    };
    /**
     *
     * @param inputs amount of neurons in input layer
     * @param outputs amount of labels/neurons in output layer
     */
    DataLoader(size_t inputs, size_t outputs) : in_(inputs), out_(outputs){}
    /**
     * @brief returns entire train dataset
     */
    const std::vector<std::pair<Matrix, Matrix>>& Data() const noexcept{return data_;}
    /**
     * @brief loads dataset(entire dataset)
     * @param mode kTrain for train dataset, kTest for test dataset
     * @param shuffle shuffle dataset
     */
    void FileToData(enum Mode mode, bool shuffle = false);
    /**
     * @brief takes sample from dataset
     * @param batch_size amount of samples\n
     * @param start_from start from x input in dataset
     * @param shuffle sample
     */
    std::vector<std::pair<Matrix, Matrix>> CreateSample(size_t batch_size = 125, size_t start_from = 0, bool shuffle = false);
    /**
     * @brief Get maximum possible amount of tests from dataset
     */
    size_t MaximumTests() {return data_.size();}
private:
    size_t in_;
    size_t out_;
    std::vector<std::pair<Matrix, Matrix>> data_;


};
}
#endif //MODELLOADER_H_
