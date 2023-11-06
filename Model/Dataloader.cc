#include "Dataloader.h"

namespace s21 {

std::vector<std::pair<std::vector<double>, std::vector<double>>>
DataLoader::CreateSample(size_t batch_size) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> sample;

    size_t batch = std::min(batch_size, train_samples_) / out_;
    size_t extra = 0;

    for(auto&[label, data] : data_){

        auto b = std::min(batch + extra, data.size());
        if(batch > data.size())
            extra +=  batch - data_.size();
        else if(batch < data.size())
            extra -= data.size() - batch;

        mtx_.lock();
        std::shuffle(data.begin(), data.end(), gen_);
        for(size_t i = 0; i < b; ++i){
            std::vector<double> ideal(out_, 0);
            ideal[label] = 1;
            std::vector<double> input(data[i]);
            sample.emplace_back(std::move(ideal), std::move(input));
        }

        mtx_.unlock();
    }
    std::shuffle(sample.begin(), sample.end(), gen_);

    return sample;
}

void DataLoader::FileToData(const char *filepath, Mode mode) {
  //  around 3350 samples per letter in initial train datasets
  std::ifstream file(filepath);
  if (!file)
    throw std::invalid_argument(
        "Dataloader: Specified file path doesn't exist");

  mode == kTrain ? FileToMap(file) : FileToVector(file);

  file.close();
}

void DataLoader::FileToMap(std::ifstream &file) {
  data_.clear();
  train_samples_ = 0;
  size_t index;
  char trash_comma;
  std::string str;

  while (std::getline(file, str)) {
      ++train_samples_;
    std::istringstream strstream(str);
    strstream >> index >> trash_comma;
    data_[--index].emplace_back(in_, 0);

    auto& sample = data_[index].back();
    for (size_t i = 0; i < in_; ++i) {
      strstream >> sample[i] >> trash_comma;
      sample[i] = sample[i] ? 1.0 : 0.0;
    }
  }
}

void DataLoader::FileToVector(std::ifstream &file) {
  test_data_.clear();
  size_t index;
  char trash_comma;
  std::string str;

  while (std::getline(file, str)) {
    std::istringstream strstream(str);
    strstream >> index >> trash_comma;
    std::vector<double> ideal(out_, 0);
    ideal[--index] = 1.0;
    test_data_.emplace_back(std::move(ideal), std::vector<double>(in_, 0));

    for (size_t i = 0; i < in_; ++i) {
      strstream >> test_data_.back().second[i] >> trash_comma;
      test_data_.back().second[i] = test_data_.back().second[i] ? 1.0 : 0.0;
    }
  }
}

}  // namespace s21
