#include "TrainingGround.h"

namespace s21{
    TrainingGround::TrainingGround(TrainingConfig& schel, DataLoader& d) : schedule_(schel), dl_(d){
        EnsureConfiguration();
        schedule_.load && !schedule_.load_path.empty() ? LoadPerceptrons() : CreatePerceptrons();
    }


    TrainingGround::~TrainingGround() {
        for(auto aboba: abobas_)
            delete aboba;
    }


    void TrainingGround::Train() {

        TrainPerceptrons();
        Test();

        if(!schedule_.save_path.empty() && (schedule_.save || schedule_.log)) {

            if(schedule_.save) {
                size_t best = FindTheBestOne();
                Save(best);
            }

            if(schedule_.log) {
                SaveLog();
            }
        }

    }


    void TrainingGround::LoadPerceptrons() {

        for (const auto &s: schedule_.load_path) {
            std::fstream file(s, std::ios_base::in);
            if (file) {
                size_t model_t;
                file >> model_t;

                abobas_.emplace_back(ConstructModel(static_cast<MLPCore::MLPType>(model_t), &dl_));

                file >> *abobas_.back();

                file.close();
            } else {
                throw std::invalid_argument("TrainingGround Constructor: load:"
                                            "specified file doesn't exist");
            }
        }

        for (size_t p = schedule_.load_path.size(); p < schedule_.perceptron_counter; ++p)
            abobas_.emplace_back(abobas_.back());

    }


    void TrainingGround::CreatePerceptrons() {

        for (size_t i = 0; i < schedule_.perceptron_counter; ++i) {

            if (dl_.Inputs() != schedule_.topologies[i].front() || dl_.Outputs() != schedule_.topologies[i].back()) {
                throw std::logic_error("TrainingGround Constructor:"
                                       "Inputs and outputs of perceptron must correspond to ins and outs of "
                                       "dataloader");
            }

            abobas_.emplace_back(ConstructModel(schedule_.mlp_types[i],
                                                schedule_.topologies[i], &dl_, schedule_.activation_functions[i]));

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

        FillMissingConfigurations();

    }


    void TrainingGround::FillMissingConfigurations(){

        while (schedule_.topologies.size() < schedule_.perceptron_counter)
            schedule_.topologies.emplace_back(schedule_.topologies.back());
        while (schedule_.mlp_types.size() < schedule_.perceptron_counter)
            schedule_.mlp_types.emplace_back(schedule_.mlp_types.back());
        while (schedule_.epochs.size() < schedule_.perceptron_counter)
            schedule_.epochs.emplace_back(schedule_.epochs.back());
        while (schedule_.activation_functions.size() < schedule_.perceptron_counter)
            schedule_.activation_functions.emplace_back(schedule_.activation_functions.back());
        while (schedule_.learning_rates.size() < schedule_.perceptron_counter)
            schedule_.learning_rates.emplace_back(schedule_.learning_rates.back());
        while (schedule_.learning_rate_reductions.size() < schedule_.perceptron_counter)
            schedule_.learning_rate_reductions.emplace_back(schedule_.learning_rate_reductions.back());
        while (schedule_.learning_rate_reduction_frequencies.size() < schedule_.perceptron_counter)
            schedule_.learning_rate_reduction_frequencies.emplace_back(schedule_.learning_rate_reduction_frequencies.back());
        while (schedule_.batch_sizes.size() < schedule_.perceptron_counter)
            schedule_.batch_sizes.emplace_back(schedule_.batch_sizes.back());

    }


    void TrainingGround::TrainPerceptrons() {

        std::vector<std::thread> they_learn;
        size_t counter = 0;

        for(auto & aboba : abobas_){

            size_t epochs = schedule_.epochs[counter];
            size_t batch_size = schedule_.batch_sizes[counter];
            double learning_rate = schedule_.learning_rates[counter];
            double learning_rate_reduction = schedule_.learning_rate_reductions[counter];
            size_t reduction_frequency = schedule_.learning_rate_reduction_frequencies[counter];
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
    void TrainingGround::Test() {

        std::vector<std::thread> they_learn;

        for(auto &aboba : abobas_){
            size_t test_batch_size = schedule_.test_batch_size;
            auto functor = [&aboba, test_batch_size](){ aboba->Test(test_batch_size); };
            they_learn.emplace_back(std::move(functor));
        }

        for(auto & t : they_learn)
            t.join();

    }

    size_t TrainingGround::FindTheBestOne(){

        size_t best_ind = 0;
        double correct_ans = abobas_[0]->Accuracy();

        for (size_t i = 1; i < abobas_.size(); ++i){
            if(correct_ans < abobas_[i]->Accuracy()){
                correct_ans = abobas_[i]->Accuracy();
                best_ind = i;
            }
        }

        return best_ind;
    }

    void TrainingGround::Save(size_t index){

        FixSaveLocation();

        std::string save_to(schedule_.save_path + abobas_[index]->ActivationFunctionName() + "_");

        for(const auto & t : abobas_[index]->Topology())
            save_to += std::to_string(t) + "_";

        save_to += "correctly_passed_" + std::to_string(abobas_[index]->Accuracy() * 100) +
                "%_of_" + std::to_string(std::min(schedule_.test_batch_size, dl_.MaximumTestSamples())) + "tests.txt";

        std::fstream file(save_to, std::ios_base::out);

        if(!file)
            throw std::logic_error("TraningGround: Save: specified path doesn't exist");

        file << *abobas_[index];
        file.close();

    }

    void TrainingGround::SaveLog(){

        FixSaveLocation();

        std::string save_to(schedule_.save_path + "log.txt");
        std::fstream file(save_to, std::ios_base::out);

        if(!file)
            throw std::logic_error("TraningGround: Save log: specified path doesn't exist");

        for(size_t p = 0 ; p < schedule_.perceptron_counter; ++p){

            std::string model = schedule_.mlp_types[p] == MLPCore::MLPType::kMatrix ? "matrix" : "graph";

            file << "MLP number " << p << " of " << model << " model type with activation function "
            << schedule_.activation_functions[p] <<   " and topology of: ";

            for(const auto& t : abobas_[p]->Topology())
                file << t << " ";
            file << std::endl << "Performance over " <<
            schedule_.epochs[p] << " epochs:" << std::endl;

            double c_lr = schedule_.learning_rates[p];

            for(size_t e = 0; e < schedule_.epochs[p]; ++e){

                file << "epoch: " << e << "\tlearning rate: " << c_lr << "\taverage error: " <<
                abobas_[p]->OutputError()[e] << std::endl;

                if(schedule_.learning_rate_reduction_frequencies[p]
                && !((e + 1) % schedule_.learning_rate_reduction_frequencies[p]))
                    c_lr -= schedule_.learning_rate_reductions[p];

            }
            file << "It took a total of " << std::to_string(abobas_[p]->TrainRuntime().count()) << " seconds" << std::endl <<
            "Running " << std::min(schedule_.test_batch_size, dl_.MaximumTestSamples()) <<
            " tests aboba correctly guessed " << abobas_[p]->Accuracy() * 100
            << "%" << std::endl << "It took a total of "
            << std::to_string(abobas_[p]->TestRuntime().count()) << " seconds" <<std::endl;

            for(size_t l = 0; l < dl_.Outputs(); ++l){
                if(std::isnan(abobas_[p]->Recall()[l]))
                    file << "Label " << l << " is not present in test set" << std::endl;
                else {
                    file << "For label " << l << "\tprecision is " << abobas_[p]->Precision()[l] <<
                         "\trecall is " << abobas_[p]->Recall()[l] << "\tf1 score is " << abobas_[p]->F1Score()[l]
                         << std::endl;
                }
            }
            std::cout << std::endl << std::endl;
        }

    }

    void TrainingGround::FixSaveLocation(){
        size_t fix_default = schedule_.save_path.find("TrainingGround.h");
        if(fix_default != std::string::npos)
            schedule_.save_path.erase(fix_default);
    }

    void TrainingGround::SetSavePath(const char * filepath) { schedule_.save_path = filepath; }
    void TrainingGround::SetTestBatchSize(const size_t size) { schedule_.test_batch_size = size; }
    void TrainingGround::SetTrainBatchSize(const size_t MLPindex, const size_t size) {
        if(MLPindex > schedule_.perceptron_counter)
            throw std::out_of_range("SetTrainBatchSize: index out of range");
        schedule_.batch_sizes[MLPindex] = size;
    }
    void TrainingGround::SetLearningRate(const size_t MLPindex, const double lr) {
        if(MLPindex > schedule_.perceptron_counter)
            throw std::out_of_range("SetLearningRate: index out of range");
        schedule_.learning_rates[MLPindex] = lr;
    }
    void TrainingGround::SetLearningRateReduction(const size_t MLPindex, const double reduction){
        if(MLPindex > schedule_.perceptron_counter)
            throw std::out_of_range("SetLearningRateReduction: index out of range");
        schedule_.learning_rate_reductions[MLPindex] = reduction;
    }
    void TrainingGround::SetLearningRateReductionFrequency(const size_t MLPindex, const size_t frequency){
        if(MLPindex > schedule_.perceptron_counter)
            throw std::out_of_range("SetLearningRateReduction: index out of range");
        schedule_.learning_rate_reduction_frequencies[MLPindex] = frequency;
    }
    void TrainingGround::SetEpochs(const size_t MLPindex, const size_t epochs){
        if(MLPindex > schedule_.perceptron_counter)
            throw std::out_of_range("SetEpochs: index out of range");
        schedule_.epochs[MLPindex] = epochs;
    }



}