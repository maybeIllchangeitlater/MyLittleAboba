#include "TrainingGround.h"

namespace s21{
    TrainingGround::TrainingGround(TrainingConfig& schel, DataLoader& d) : schedule_(schel), dl_(d){
        schedule_.load && !schedule_.path_to_perceptrons.empty() ? LoadPerceptrons() : CreatePerceptrons();
    }


    void TrainingGround::Train() {

        std::vector<std::thread> they_learn;

        TrainPerceptrons(they_learn);
        they_learn.clear();

        TestPerceptrons(they_learn);
        they_learn.clear();

        SaveAccuracy();
        if(schedule_.save && !schedule_.winner_savepath.empty())
            SaveTheBestOne();

    }

    void TrainingGround::LoadPerceptrons() {

        for (const auto &s: schedule_.path_to_perceptrons) {
            std::fstream file(s, std::ios_base::in);
            if (file) {
                abobas_.emplace_back(&dl_);
                file >> abobas_.back();
                file.close();
            } else {
                throw std::invalid_argument("TrainingGround Constructor: load:"
                                            "specified file doesn't exist");
            }
        }

        for (size_t p = schedule_.path_to_perceptrons.size(); p < schedule_.perceptron_counter; ++p)
            abobas_.emplace_back(abobas_.back());

    }

    void TrainingGround::CreatePerceptrons() {

        if(!schedule_.perceptron_counter) schedule_.perceptron_counter = 1;

        for (size_t i = 0; i < schedule_.perceptron_counter; ++i) {
            if (dl_.Inputs() != schedule_.topologies[i].front() || dl_.Outputs() != schedule_.topologies[i].back())
                throw std::logic_error("TrainingGround Constructor:"
                                       "Inputs and outputs of perceptron must correspond to ins and outs of "
                                       "dataloader");

            abobas_.emplace_back(schedule_.topologies[i], &dl_);
        }

    }

    void TrainingGround::TrainPerceptrons(std::vector<std::thread>& they_learn) {

        size_t counter = 0;

        for(MLP &aboba : abobas_){ //alternatively [std::min(size -1, counter)], its a mess either ways
            size_t epochs = counter < schedule_.epochs.size()
                    ? schedule_.epochs[counter]
                    : schedule_.epochs.back();
            size_t batch_size = counter < schedule_.batch_sizes.size()
                                ? schedule_.batch_sizes[counter]
                                : schedule_.batch_sizes.back();
            double learning_rate = counter < schedule_.learning_rates.size()
                                   ? schedule_.learning_rates[counter]
                                   : schedule_.learning_rates.back();
            double learning_rate_reduction = counter < schedule_.learning_rate_reductions.size()
                                            ? schedule_.learning_rate_reductions[counter]
                                            : schedule_.learning_rate_reductions.back();
            size_t reduction_frequency = counter < schedule_.learning_rate_reduction_frequencies.size()
                                         ? schedule_.learning_rate_reduction_frequencies[counter]
                                         : schedule_.learning_rate_reduction_frequencies.back();
            ++counter;

            auto functor = [&aboba, epochs, batch_size, learning_rate,
                    learning_rate_reduction, reduction_frequency](){
                aboba.GradientDescent(learning_rate, epochs, batch_size,
                                      learning_rate_reduction, reduction_frequency);};

            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }
    void TrainingGround::TestPerceptrons(std::vector<std::thread> &they_learn) {

        for(MLP &aboba : abobas_){
            auto functor = [&aboba](){ aboba.Test(); };
            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }

    size_t TrainingGround::FindTheBestOne(){

        size_t best_ind = 0;
        size_t correct_ans = correctness_counter.emplace_back(abobas_[0].CorrectAnswers());

        for (size_t i = 1; i < abobas_.size(); ++i){
            correctness_counter.emplace_back(abobas_[i].CorrectAnswers());
            if(correct_ans < correctness_counter.back()){
                correct_ans = correctness_counter.back();
                best_ind = i;
            }
        }

        return best_ind;
    }

    void TrainingGround::SaveTheBestOne(){

        size_t best = FindTheBestOne();

        for(const auto & l : abobas_[best].GetLayers())
            schedule_.winner_savepath += std::to_string(l.activated_outputs_.Size()) + "_";

        schedule_.winner_savepath += "correctly_passed_" + std::to_string(abobas_[best].CorrectAnswers());
        std::string save_log_to(schedule_.winner_savepath + "log.txt");
        save_log_to += "tests.txt";

        std::fstream file(schedule_.winner_savepath, std::ios_base::out);
        std::fstream log_file(save_log_to, std::ios_base::out);
        if(!file || !log_file) {
            file.close();
            log_file.close();
            throw std::logic_error("TraningGround: Save: specified path doesn't exist");
        }
        file << abobas_[best];
        file.close();


        log_file.close();

    }

    void TrainingGround::SaveAccuracy(){
        for(const auto& aboba : abobas_){
            accuracy.emplace_back();
            for(const auto& a : aboba.GetAccuracy())
                accuracy.back().emplace_back(a);
        }
    }


}