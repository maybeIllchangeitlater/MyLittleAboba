#include "dataloader.h"
#include <fstream>
#include <sstream>

namespace s21{

    std::vector<std::pair<S21Matrix, S21Matrix>> DataLoader::CreateSample(size_t batch_size, size_t start_from){

        std::vector<std::pair<S21Matrix, S21Matrix>> sample;
        size_t finish = start_from + batch_size;

        for(size_t i = start_from; i < finish; ++i)
                sample.push_back(data_[i]);
        return sample;
}

void DataLoader::FileToData(Mode mode)
{
    data_.clear();
    size_t index;
    char trash_comma;
    std::string str;
    std::ifstream file(mode == kTrain ? "/Users/monke/Biba/emnist-letters/emnist-letters-train.csv"
    : "/Users/monke/Biba/emnist-letters/emnist-letters-test.csv"); // around 3350 samples per letter in initial datasets
    while(std::getline(file, str)){
       std::istringstream strstream(str);
       strstream >> index >> trash_comma;
       S21Matrix ideal(1, out_);
       ideal(0, --index) = 1.0;
        data_.emplace_back(ideal, S21Matrix(1, in_));
        for(int i = 0; i < in_; ++i) {
            strstream >> data_.back().second(0, i) >> trash_comma;
            data_.back().second(0, i) = data_.back().second(0, i) ? 1.0 : 0.0;
        }
    }
    file.close();
}


}

