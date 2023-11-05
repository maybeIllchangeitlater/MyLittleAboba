#ifndef MULTILAYERABOBATRON_MODEL_MLPINTERAFACE_H
#define MULTILAYERABOBATRON_MODEL_MLPINTERAFACE_H
#include <string>
#include <vector>

namespace s21{
    class MLPInterface{ //actually just a normal abstract class
    public:

        virtual ~MLPInterface() = default;
        /**
         * @brief Get that error down
         * @param lr learning rate. defaulted to 0.03
         * @param epochs to go through. defaulted to 5
         * @param batch_size defaulted to entire dataset
         * @param lr_reduction reduce lr by. defaulted to 0.0
         * @param reduction_frequency per how many epochs to reduce lr. defaulted to 0 (never)
         */
        virtual void GradientDescent(double lr = 0.1, size_t epochs = 5, size_t batch_size = SIZE_T_MAX,
                                     double lr_reduction = 0.0, size_t reduction_frequency = 0) = 0;
        /**
         * @brief parse input and guess a label
         */
        virtual size_t Predict(const std::vector<double>& in) = 0;
        /**
         * @brief run the tests
         * @return amount of correct guesses
         */
        virtual size_t Test() = 0;
        /**
         * @brief returns topology of mlp instance
         */
        virtual std::vector<size_t> Topology() = 0;

        const std::string& ActivationFunctionName() { return activation_function_name_; }
        /**
         * @return how many tests did MatrixMLP pass last test run
         */
        size_t CorrectAnswers() const noexcept { return correct_test_answers_; };
        /**
         *
         * @return average error over each epoch
         */
        const std::vector<double>& GetAccuracy() const noexcept { return average_error_; }
        /**
         * @brief save MLP
         */
        friend std::ostream &operator<<(std::ostream &out, const MLPInterface &other){
            other.Out(out);
            return out;
        }
        /**
         * @brief load MLP
         */
        friend std::istream &operator>>(std::istream &in, MLPInterface &other){
            other.In(in);
            return in;
        }

    protected:
        ///for <<
        virtual void Out(std::ostream &out) const = 0;
        ///for >>
        virtual void In(std::istream &in) = 0;

        size_t correct_test_answers_;

        std::string activation_function_name_;

        std::vector<double> average_error_;
    };
}

#endif //MULTILAYERABOBATRON_MODEL_MLPINTERAFACE_H
