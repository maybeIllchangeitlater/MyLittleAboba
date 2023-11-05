#include "Dataloader.h"


namespace s21{

    std::vector<std::pair<Mx, Mx>> DataLoader::CreateSample(size_t batch_size, size_t start_from, Mode mode, bool shuffle){

        auto & data = mode == kTrain ? data_ : test_data_;
        std::vector<std::pair<Mx, Mx>> sample;
        size_t finish = start_from + batch_size;

        for(size_t i = start_from; i < finish; ++i)
                sample.push_back(data[i]);

        if(shuffle)
            std::shuffle(sample.begin(), sample.end(), gen_);

        return sample;
}

void DataLoader::FileToData(const char * filepath, Mode mode, bool shuffle)
{
//  around 3350 samples per letter in initial train datasets
    std::ifstream file(filepath);
    size_t index;
    char trash_comma;
    std::string str;

    auto & data = mode == kTrain ? data_ : test_data_;
    if(mode == kTrain)
        data_.clear();
    else
        test_data_.clear();

    while(std::getline(file, str)){
       std::istringstream strstream(str);
       strstream >> index >> trash_comma;
       Mx ideal(1, out_);
       ideal(0, --index) = 1.0;
       data.emplace_back(ideal, Mx(1, in_));

        for(size_t i = 0; i < in_; ++i) {
            strstream >> data.back().second(0, i) >> trash_comma;
            data.back().second(0, i) = data.back().second(0, i) ? 1.0 : 0.0;
        }

    }

    if(shuffle)
        std::shuffle(data.begin(), data.end(), gen_);

    file.close();

}


}

