#include "dataloader.h"


namespace s21{

    std::vector<std::pair<S21Matrix, S21Matrix>> DataLoader::CreateSample(size_t batch_size, size_t start_from, Mode mode, bool shuffle){
        auto & data = mode == kTrain ? data_ : test_data_;
        std::vector<std::pair<S21Matrix, S21Matrix>> sample;
        size_t finish = start_from + batch_size;

        for(size_t i = start_from; i < finish; ++i)
                sample.push_back(data[i]);
        if(shuffle){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(sample.begin(), sample.end(), gen);
        }
        return sample;
}

void DataLoader::FileToData(const char * filepath, Mode mode, bool shuffle)
{
//    data_.clear();
    std::ifstream file(filepath);
    size_t index;
    char trash_comma;
    std::string str;
    auto & data = mode == kTrain ? data_ : test_data_;
    if(mode == kTrain)
        data_.clear();
    else
        test_data_.clear();
//    std::ifstream file(mode == kTrain ? "/Users/monke/Biba/emnist-letters/emnist-letters-train.csv"
//    : "/Users/monke/Biba/emnist-letters/emnist-letters-test.csv"); // around 3350 samples per letter in initial datasets
    while(std::getline(file, str)){
       std::istringstream strstream(str);
       strstream >> index >> trash_comma;
       S21Matrix ideal(1, out_);
       ideal(0, --index) = 1.0;
        data.emplace_back(ideal, S21Matrix(1, in_));
        for(int i = 0; i < in_; ++i) {
            strstream >> data.back().second(0, i) >> trash_comma;
            data.back().second(0, i) = data.back().second(0, i) ? 1.0 : 0.0;
        }
    }
    if(shuffle){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(data.begin(), data.end(), gen);
    }
    file.close();
}


}

