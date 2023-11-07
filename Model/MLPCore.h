#ifndef MULTILAYERABOBATRON_MODEL_MLPCORE_H
#define MULTILAYERABOBATRON_MODEL_MLPCORE_H
#include <string>
#include <vector>

namespace s21{
    class MLPCore{ //abstract
    public:
        enum MLPType{
            kMatrix,
            kGraph
        };

        virtual ~MLPCore() = default;
        /**
         * @brief Get that error down
         * @param lr learning rate. defaulted to 0.03
         * @param epochs to go through. defaulted to 5
         * @param batch_size defaulted to entire dataset
         * @param lr_reduction reduce lr by. defaulted to 0.0
         * @param reduction_frequency per how many epochs to reduce lr. defaulted to 0 (never)
         */
        virtual void GradientDescent(double lr = 0.03, size_t epochs = 5, size_t batch_size = SIZE_T_MAX,
                                     double lr_reduction = 0.0, size_t reduction_frequency = 0) = 0;
        /**
         * @brief parse input and guess a label
         */
        virtual size_t Predict(const std::vector<double>& in) = 0;
        /**
         * @brief run the batch_size amount of tests and find model's precision, accuracy,\n
         * recall and F1
         */
        virtual void Test(size_t batch_size = SIZE_T_MAX) = 0;
        /**
         * @brief returns topology of mlp instance
         */
        virtual std::vector<size_t> Topology() = 0;

        const std::string& ActivationFunctionName() const noexcept { return activation_function_name_; }
        /**
         * @brief correct / total answers
         */
        const double& Accuracy() const noexcept { return accuracy_; }
        /**
         * @brief true positives / total positives per label\n
         * if label wasn't present in test sample will return INF
         */
        const std::vector<double>& Precision() const noexcept { return precision_; }
        /**
         * @brief true positives / all positives (tp / tp + fn) per label\n
         * if label wasn't present in test sample will return INF
         */
         const std::vector<double>& Recall() const noexcept { return recall_; }
         /**
          * @brief 2 * precision * recall/ precision + recall
          */
         const std::vector<double>& F1Score() const noexcept { return f1_score_; }
        /**
         * @brief average output layer gradient per epoch
         */
        const std::vector<double>& AverageOutputGradient()const noexcept { return error_; }
        /**
         * @brief save MLP
         */
        friend std::ostream &operator<<(std::ostream &out, const MLPCore &other){
            other.Out(out);
            return out;
        }
        /**
         * @brief load MLP
         */
        friend std::istream &operator>>(std::istream &in, MLPCore &other){
            other.In(in);
            return in;
        }

    protected:
        ///for <<
        virtual void Out(std::ostream &out) const = 0;
        ///for >>
        virtual void In(std::istream &in) = 0;

        double lr_;
        std::string activation_function_name_;
        std::vector<double> error_;
        double accuracy_;
        std::vector<double> precision_;
        std::vector<double> recall_;
        std::vector<double> f1_score_;
    };
}

#endif //MULTILAYERABOBATRON_MODEL_MLPCORE_H
