#include "TrainingGround.h"

namespace s21{
    TrainingGround::TrainingGround(s21::TrainingConfig& schel, DataLoader& d) : schedule_(schel), dl_(d){
        schedule_.load && !schedule_.path_to_perceptrons.empty() ? LoadPreceptrons() : CreatePreceptrons();
    } ///work with schel sizes possibly leading to sega


    void TrainingGround::Train() {
        std::vector<std::thread> they_learn;
        TrainPreceptrons(they_learn);
        they_learn.clear();
        TestPreceptrons(they_learn);
        they_learn.clear();
        SaveTheBestOne();
    }

    void TrainingGround::LoadPreceptrons() {

        for (const auto &s: schedule_.path_to_perceptrons) {
            abobas_.emplace_back(&dl_);
            std::fstream file(s, std::ios_base::in);
            if (file) {
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

    void TrainingGround::CreatePreceptrons() {

        if(!schedule_.perceptron_counter) schedule_.perceptron_counter = 1;

        for (size_t i = 0; i < schedule_.perceptron_counter; ++i) {
            if (dl_.Inputs() != schedule_.topologies[i].front() || dl_.Outputs() != schedule_.topologies[i].back())
                throw std::logic_error("TrainingGround Constructor:"
                                       "Inputs and outputs of preceptron must correspond to ins and outs of "
                                       "dataloader");

            abobas_.emplace_back(schedule_.topologies[i], &dl_);
        }

    }

    void TrainingGround::TrainPreceptrons(std::vector<std::thread>& they_learn) {

        size_t counter = 0;

        for(MLP &aboba : abobas_){
            size_t epochs = schedule_.epochs[counter];
            size_t batch_size = schedule_.batch_sizes[counter];
            size_t iterations_for_batch = schedule_.batch_iterations[counter];
            double learning_rate = schedule_.learning_rates[counter].first;
            double learning_rate_reduction = schedule_.learning_rates[counter].second.first;
            size_t reduction_frequency = schedule_.learning_rates[counter].second.second;
            ++counter;

            auto functor = [&aboba, epochs, iterations_for_batch, batch_size, learning_rate,
                    learning_rate_reduction, reduction_frequency](){
                aboba.GradientDescent(learning_rate, epochs, iterations_for_batch, batch_size,
                                      learning_rate_reduction, reduction_frequency);};

            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }
    void TrainingGround::TestPreceptrons(std::vector<std::thread> &they_learn) {

        for(MLP &aboba : abobas_){
            auto functor = [&aboba](){ aboba.Test(); };
            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }

    size_t TrainingGround::FindTheBestOne(){

        size_t best_ind = 0;
        size_t correct_anss = correctness_counter.emplace_back(abobas_[0].CorrectAnswers());

        for (size_t i = 1; i < abobas_.size(); ++i){
            correctness_counter.emplace_back(abobas_[i].CorrectAnswers());
            if(correct_anss < correctness_counter.back()){
                correct_anss = correctness_counter.back();
                best_ind = i;
            }
        }

        return best_ind;
    }

    void TrainingGround::SaveTheBestOne(){

        size_t best = FindTheBestOne();
        std::string save_to(schedule_.winner_savepath);

        for(const auto & l : abobas_[best].GetLayers())
            save_to += std::to_string(l.activated_outputs_.Size()) + "_";

        save_to += "correctly_passed_" + std::to_string(abobas_[best].CorrectAnswers()) + "tests.txt";
        std::fstream file(save_to, std::ios_base::out);

        file << abobas_[best];

        file.close();
    }



    }