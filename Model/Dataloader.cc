#include "Dataloader.h"

namespace s21 {

std::vector<std::pair<size_t, std::vector<double>>> DataLoader::CreateSample(
    size_t batch_size, Mode mode) {
  auto& dataset = mode == kTrain ? data_ : test_data_;
  std::vector<std::pair<size_t, std::vector<double>>> sample;

  size_t batch =
      std::min(batch_size, mode == kTrain ? train_samples_ : test_samples_) /
      out_;
  size_t extra = 0;

  for (auto& [label, data] : dataset) {
    auto b = std::min(batch + extra, data.size());
    if (batch > data.size())
      extra += batch - dataset.size();
    else if (batch < data.size())
      extra -= data.size() - batch;

    mutex_.lock();
    std::shuffle(data.begin(), data.end(), gen_);
    for (size_t i = 0; i < b; ++i) {
      std::vector<double> input(data[i]);
      sample.emplace_back(label, std::move(input));
    }

    mutex_.unlock();
  }
  std::shuffle(sample.begin(), sample.end(), gen_);

  return sample;
}

std::vector<double> DataLoader::PicToData(QPixmap& pixmap) {
  std::vector<double> data_sample;
  data_sample.reserve(in_);
  size_t w = std::sqrt(in_);
  QImage image(pixmap.scaled(w, w).toImage());
  for (size_t i = 0; i < w; ++i) {
    for (size_t j = 0; j < w; ++j) {
      data_sample.emplace_back(qRed(image.pixel(i, j)) ? 0 : 1);  // switch 0 1
    }
  }
  return data_sample;
}

void DataLoader::FileToData(const char* filepath, Mode mode) {
  //  around 3350 samples per letter in initial train datasets
  std::ifstream file(filepath);
  if (!file)
    throw std::invalid_argument(
        "Dataloader: Specified file path doesn't exist");

  auto& data = mode == kTrain ? data_ : test_data_;
  auto& samples = mode == kTrain ? train_samples_ : test_samples_;
  data.clear();
  samples = 0;

  size_t index;
  char trash_comma;
  std::string str;

  while (std::getline(file, str)) {
    ++samples;
    std::istringstream strstream(str);
    strstream >> index >> trash_comma;
    data[--index].emplace_back(in_, 0);

    auto& sample = data[index].back();
    for (size_t i = 0; i < in_; ++i) {
      strstream >> sample[i] >> trash_comma;
      sample[i] = sample[i] ? 1.0 : 0.0;
    }
  }

  file.close();
}

}  // namespace s21
