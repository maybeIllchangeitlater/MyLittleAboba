#include "dataloader.h"
#include <fstream>
#include <sstream>

namespace s21{

void DataLoader::DisplayMatrices() {
    for(const auto&[k, v] : data_){
        std::cout << "Letter is: " << static_cast<char>('a' + k) << std::endl;
        for(const auto &val: v){
            for(int i = 0; i < 28; ++i){
                for(int j = 0; j < 28; ++j){
                    std::cout << val[i * 28 + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void DataLoader::CreateSample(unsigned tests_per_letter){
    sample_.clear();
    for(size_t i = 0; i < tests_per_letter; ++i){
        for(size_t j = 0; j < 26; ++j){
            sample_[j].push_back(data_[j][i]);
        }
    }
}

void DataLoader::FileToData(Mode mode)
{
    data_.clear();
    unsigned index;
    char trash_comma;
    std::string str;
    std::ifstream file(mode == kTrain ? "/Users/monke/Biba/emnist-letters/emnist-letters-train.csv" : "/Users/monke/Biba/emnist-letters/emnist-letters-test.csv",
                       std::ios::in);
    std::vector<double> tmp;
    tmp.resize(784);
    while(std::getline(file, str)){
       std::istringstream strstream(str);
       strstream >> index >> trash_comma;
        for(int i = 0; i < 784; ++i){
                strstream >> tmp[i] >> trash_comma;
                tmp[i] = !tmp[i] ? 0.0 : 1.0;
            }
        data_[--index].push_back(tmp);
    }
    file.close(); // probably not needed because of destructor
}



}

