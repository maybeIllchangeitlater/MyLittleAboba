#ifndef MODELLOADER_H_
#define MODELLOADER_H_

#include <unordered_map>
#include <iostream>

namespace s21{
class DataLoader{
public:
    enum Mode{
      kTrain,
        kTest
    };
    std::unordered_map<unsigned, std::vector<std::vector<double>>>& Data() noexcept{return data_;}
    std::unordered_map<unsigned, std::vector<std::vector<double>>>& Sample() noexcept{return sample_;}
    void FileToData(enum Mode mode);
    void DisplayMatrices();
    void CreateSample(unsigned int tests_per_letter);
private:
    unsigned counter = 0;
    std::unordered_map<unsigned, std::vector<std::vector<double>>> data_;
    std::unordered_map<unsigned, std::vector<std::vector<double>>> sample_;

};
}
#endif //MODELLOADER_H_
