#include "TrainingGround.h"

namespace s21{
    TrainingGround::TrainingGround(s21::TrainingConfig& schel, DataLoader& d) : schelude(schel), dl(d){
        if(schelude.load){
            for(const auto& s : schelude.path_to_preceptrons){
                abobas.emplace_back(&dl);
                std::fstream file(s, std::ios_base::in);
                file >> abobas.back();
            }
            if(schelude.path_to_preceptrons.size() == 1 && schelude.copy){
                for(size_t i = 0; i < schelude.copy; ++i){
                    abobas.emplace_back(abobas.back());
                }
            }
        } else {
            for (size_t i = 0; i < schelude.preceptron_counter; ++i) {
                if (dl.Inputs() != schelude.topologies[i].front() || dl.Outputs() != schelude.topologies[i].back())
                    throw std::logic_error("TrainingGround Constructor:"
                                           "Inputs and outputs of preceptron must correspond to ins and outs of "
                                           "dataloader");
                abobas.emplace_back(schelude.topologies[i], &dl, schelude.learning_rates[i].first);
            }
        }
    }
    void TrainingGround::Train() {
        std::vector<std::thread> they_learn;
        size_t counter = 0;
        for(MLP &aboba : abobas){
            size_t epochs = schelude.epochs[counter];
            size_t batch_size = schelude.batch_sizes[counter];
            size_t iterations_for_batch = schelude.batch_iterations[counter];
            double learning_rate_reduction = schelude.learning_rates[counter].second.first;
            size_t reduction_frequency = schelude.learning_rates[counter].second.second;
            ++counter;
            auto functor = [&aboba, epochs, iterations_for_batch, batch_size,
                learning_rate_reduction, reduction_frequency](){
                aboba.GradientDescent(epochs, iterations_for_batch, batch_size,
                                      learning_rate_reduction, reduction_frequency);};
            they_learn.emplace_back(functor);
        }
        for(auto & t : they_learn){
            t.join();
        }
        they_learn.clear();
        for(MLP &aboba : abobas){
            auto functor = [&aboba](){ aboba.Test(); };
            they_learn.emplace_back(functor);
        }
        for(auto & t : they_learn){
            t.join();
        }
        they_learn.clear();
        size_t best_ind = 0;
        size_t correct_anss = correctness_counter.emplace_back(abobas[0].CorrectAnswers());
        for (size_t i = 1; i < abobas.size(); ++i){
            correctness_counter.emplace_back(abobas[i].CorrectAnswers());
            if(correct_anss < correctness_counter.back()){
                correct_anss = correctness_counter.back();
                best_ind = i;
            }
        }
        std::string save_to(schelude.winner_savepath);
        for(const auto& t : schelude.topologies[best_ind]){
            save_to += std::to_string(t) + "_";
        }
        save_to += "correctly_passed_" + std::to_string(abobas[best_ind].CorrectAnswers()) + "tests.txt";
        std::fstream file(save_to, std::ios_base::out);

        file << abobas[best_ind];

        file.close();
    }

}