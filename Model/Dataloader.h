#ifndef MULTILAYERABOBATRON_MODEL_DATALOADER_H_
#define MULTILAYERABOBATRON_MODEL_DATALOADER_H_

#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

namespace s21{
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
    const std::unordered_map<size_t, std::vector<std::vector<double>>>& Data() const noexcept { return data_; }
     /**
      * @brief returns entire test dataset
      */
     const std::vector<std::pair<std::vector<double>, std::vector<double>>>& TestData() const noexcept {return test_data_;}
    /**
     * @brief loads dataset(entire dataset)
     * @param filepath path to dataset
     * @param mode is dataset for testing kTest or training kTrain
     */
    void FileToData(const char * filepath, Mode mode);
    /**
     * @brief takes sample from train dataset
     * @param batch_size size of sample\n
     */
    std::vector<std::pair<std::vector<double>, std::vector<double>>> CreateSample
    (size_t batch_size = SIZE_T_MAX);
    /**
     * @brief Get maximum possible amount of samples from learning dataset
     */
    size_t MaximumTests() const noexcept {return train_samples_;}
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
        void FileToMap(std::ifstream &file);
    void FileToVector(std::ifstream &file);
    size_t train_samples_;
        size_t in_;
        size_t out_;
        std::unordered_map<size_t, std::vector<std::vector<double>>> data_;
        std::vector<std::pair<std::vector<double>, std::vector<double>>> test_data_;
        std::mutex mtx_;
        std::mt19937 gen_;

};
}
#endif //MULTILAYERABOBATRON_MODEL_DATALOADER_H_
