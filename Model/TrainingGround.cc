#include "TrainingGround.h"

namespace s21{
    TrainingGround::TrainingGround(TrainingConfig& schel, DataLoader& d) : schedule_(schel), dl_(d){
        EnsureConfiguration();
        schedule_.load && !schedule_.path_to_perceptrons.empty() ? LoadPerceptrons() : CreatePerceptrons();
    }


    TrainingGround::~TrainingGround() {
        for(auto& a: abobas_)
            delete a;
    }


    void TrainingGround::Train() {

        std::vector<std::thread> they_learn;

        TrainPerceptrons(they_learn);
        they_learn.clear();

        TestPerceptrons(they_learn);
        they_learn.clear();

        FindAccuracy();

        size_t best = FindTheBestOne();
        if(schedule_.save && !schedule_.winner_savepath.empty())
            SaveTheBestOne(best);

    }


    void TrainingGround::LoadPerceptrons() {

        for (const auto &s: schedule_.path_to_perceptrons) {
            std::fstream file(s, std::ios_base::in);
            if (file) {
                size_t model_t;
                file >> model_t;

                abobas_.emplace_back(ConstructModel(static_cast<TrainingConfig::MLPType>(model_t), &dl_));

                file >> *abobas_.back();

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


        for (size_t i = 0; i < schedule_.perceptron_counter; ++i) {

            auto & topology = i < schedule_.topologies.size()
                    ? schedule_.topologies[i]
                    : schedule_.topologies.back();

            if (dl_.Inputs() != topology.front() || dl_.Outputs() != topology.back()) {
                throw std::logic_error("TrainingGround Constructor:"
                                       "Inputs and outputs of perceptron must correspond to ins and outs of "
                                       "dataloader");
            }

            auto& activation = i < schedule_.activation_functions.size()
                                        ? schedule_.activation_functions[i]
                                        : schedule_.activation_functions.back();
            auto& model_type = i < schedule_.mlp_types.size()
                                       ? schedule_.mlp_types[i]
                                       : schedule_.mlp_types.back();

            abobas_.emplace_back(ConstructModel(model_type, topology, &dl_, activation));

        }

    }


    void TrainingGround::EnsureConfiguration() {

        if(schedule_.topologies.empty() && !schedule_.load)
            throw std::logic_error("TrainingGround Constructor: Specify at least one topology");

        if(!schedule_.perceptron_counter)
            schedule_.perceptron_counter = std::thread::hardware_concurrency();
        if(schedule_.mlp_types.empty())
            schedule_.mlp_types.emplace_back(TrainingConfig::kDefaultMLPType);
        if(schedule_.epochs.empty())
            schedule_.epochs.emplace_back(TrainingConfig::kDefaultEpochs);
        if(schedule_.activation_functions.empty())
            schedule_.activation_functions.emplace_back(TrainingConfig::kDefaultActivator);
        if(schedule_.learning_rates.empty())
            schedule_.learning_rates.emplace_back(TrainingConfig::kDefaultLR);
        if(schedule_.learning_rate_reductions.empty())
            schedule_.learning_rate_reductions.emplace_back(TrainingConfig::kDefaultLRReductionRate);
        if(schedule_.learning_rate_reduction_frequencies.empty())
            schedule_.learning_rate_reduction_frequencies.emplace_back(TrainingConfig::kDefaultLRReductionFrequency);
        if(schedule_.batch_sizes.empty())
            schedule_.batch_sizes.emplace_back(TrainingConfig::kDefaultBatchSize);

    }


    void TrainingGround::TrainPerceptrons(std::vector<std::thread>& they_learn) {

        size_t counter = 0;

        for(auto & aboba : abobas_){ //alternatively [std::min(size -1, counter)], its a mess either ways
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
                aboba->GradientDescent(learning_rate, epochs, batch_size,
                                      learning_rate_reduction, reduction_frequency);};

            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }
    void TrainingGround::TestPerceptrons(std::vector<std::thread> &they_learn) {

        for(auto &aboba : abobas_){
            auto functor = [&aboba](){ aboba->Test(); };
            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }

    size_t TrainingGround::FindTheBestOne(){

        size_t best_ind = 0;
        size_t correct_ans = correctness_counter.emplace_back(abobas_[0]->CorrectAnswers());

        for (size_t i = 1; i < abobas_.size(); ++i){
            correctness_counter.emplace_back(abobas_[i]->CorrectAnswers());
            if(correct_ans < correctness_counter.back()){
                correct_ans = correctness_counter.back();
                best_ind = i;
            }
        }

        return best_ind;
    }

    void TrainingGround::SaveTheBestOne(size_t best){

        size_t fix_default = schedule_.winner_savepath.find("TrainingGround.h");
        if(fix_default != std::string::npos)
            schedule_.winner_savepath.erase(fix_default);

        schedule_.winner_savepath += abobas_[best]->ActivationFunctionName() + "_";

        for(const auto & t : abobas_[best]->Topology())
            schedule_.winner_savepath += std::to_string(t) + "_";

        schedule_.winner_savepath += "correctly_passed_" + std::to_string(abobas_[best]->CorrectAnswers());

        if(schedule_.log)
            SaveLog(best);

        schedule_.winner_savepath += "tests.txt" ;

        std::fstream file(schedule_.winner_savepath, std::ios_base::out);


        if(!file)
            throw std::logic_error("TraningGround: Save: specified path doesn't exist");

        file << abobas_[best];
        file.close();

    }

    void TrainingGround::SaveLog(size_t best){

        std::string save_log(schedule_.winner_savepath + "log.txt");
        std::fstream file_log(save_log, std::ios_base::out);

        if(!file_log)
            throw std::logic_error("TraningGround: Save log: specified path doesn't exist");

        for(const auto& a : abobas_[best]->GetAccuracy())
            file_log << a << " ";

    }


    void TrainingGround::FindAccuracy(){
        for(const auto& aboba : abobas_){
            accuracy.emplace_back();
            for(const auto& a : aboba->GetAccuracy())
                accuracy.back().emplace_back(a);
        }
    }


}