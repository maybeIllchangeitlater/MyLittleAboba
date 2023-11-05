#ifndef MULTILAYERABOBATRON_UTILS_MLPMATRIX_H_
#define MULTILAYERABOBATRON_UTILS_MLPMATRIX_H_

#include <cmath>
#include <cstring>  //for memcpy
#include <iostream>
#include <random>
#include <functional>

namespace s21 {
    class MLPMatrix {
    public:

        MLPMatrix() noexcept: rows_(0), cols_(0), matrix_(nullptr){}
        explicit MLPMatrix(const size_t rows, const size_t cols)  : rows_(rows), cols_(cols) {
            if(rows_ && cols_) {
                matrix_ = new double *[rows_];
                matrix_[0] = new double[rows_ * cols_]();
                for (size_t i = 1; i < rows_; i++) matrix_[i] = matrix_[0] + i * cols_;
            }else{
                matrix_ = nullptr;
            }
        }
        MLPMatrix(const size_t rows, const size_t cols, std::mt19937& generator, const double from, const double to)
                :rows_(rows), cols_(cols){
            std::uniform_real_distribution<double> dist(from, to);
            matrix_ = new double *[rows_];
            matrix_[0] = new double [rows_ * cols_];
            for(size_t j = 0; j < cols_; ++j) matrix_[0][j] = dist(generator);
            for (size_t i = 1; i < rows_; i++) {
                matrix_[i] = matrix_[0] + i * cols_;
                for(size_t j = 0; j < cols_; ++j) matrix_[i][j] = dist(generator);
            }
        }
        MLPMatrix(const MLPMatrix &other): MLPMatrix(other.rows_, other.cols_) {
            if(matrix_)
                std::memcpy(matrix_[0], other.matrix_[0], sizeof(double) * cols_ * rows_);
        }
        MLPMatrix(MLPMatrix &&other) noexcept: rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
            other.rows_ = 0;
            other.cols_ = 0;
            other.matrix_ = nullptr;
        }
        MLPMatrix &operator=(const MLPMatrix &other){
            if (this == &other) return *this;
            MLPMatrix tmp(other);
            *this = std::move(tmp);
            return *this;
        }
        MLPMatrix &operator=(MLPMatrix &&other) noexcept{
            if(&other == this) return *this;
            if(matrix_) delete[] matrix_[0];
            delete[] matrix_;
            rows_ = other.rows_, cols_ = other.cols_, matrix_ = other.matrix_;
            other.rows_ = 0, other.cols_ = 0;
            other.matrix_ = nullptr;
            return *this;
        }
        ~MLPMatrix(){
            if(matrix_) delete[] matrix_[0];
            delete[] matrix_;
        }


        MLPMatrix operator+(const MLPMatrix &other) const{
            MLPMatrix res(rows_, cols_);
            std::transform(begin(), end(), other.begin(), res.begin(), std::plus<>());
            return res;
        }
        MLPMatrix &operator+=(const MLPMatrix &other){
            std::transform(begin(), end(), other.begin(), begin(), std::plus<>());
            return *this;
        }


        MLPMatrix operator-(const MLPMatrix &other) const{
            MLPMatrix res(rows_, cols_);
            std::transform(begin(), end(), other.begin(), res.begin(), std::minus<>());
            return res;
        }
        MLPMatrix &operator-=(const MLPMatrix &other){
            std::transform(begin(), end(), other.begin(), begin(), std::minus<>());
            return *this;
        }


        MLPMatrix operator&(const MLPMatrix& other) const{
            MLPMatrix res(rows_, cols_);
            std::transform(begin(), end(), other.begin(), res.begin(), std::multiplies<>());
            return res;
        }
        MLPMatrix &operator&=(const MLPMatrix& other) {
            std::transform(begin(), end(), other.begin(), begin(), std::multiplies<>());
            return *this;
        }


        MLPMatrix operator*(const MLPMatrix &other) const{
            MLPMatrix res(rows_, other.cols_);
            for (size_t i = 0; i < rows_; i++)
                for (size_t j = 0; j < other.cols_; j++)
                    for (size_t k = 0; k < other.rows_; k++)
                        res.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
            return res;
        }
        MLPMatrix &operator*=(const MLPMatrix &other){
            *this = (*this * other);
            return *this;
        }
        MLPMatrix MulByTransposed(const MLPMatrix& other) const{
            MLPMatrix res(rows_, other.rows_);
            for (size_t i = 0; i < rows_; i++)
                for (size_t j = 0; j < other.rows_; j++)
                    for (size_t k = 0; k < cols_; k++)
                        res.matrix_[i][j] += matrix_[i][k] * other.matrix_[j][k];
            return res;
        }
        MLPMatrix MulSelfTranspose(const MLPMatrix &other) const{
            MLPMatrix res(cols_, other.cols_);
            for (size_t i = 0; i < cols_; i++)
                for (size_t j = 0; j < other.cols_; j++)
                    for (size_t k = 0; k < rows_; k++)
                        res.matrix_[i][j] += matrix_[k][i] * other.matrix_[k][j];
            return res;
        }

        MLPMatrix operator*(const double num) const{
            MLPMatrix res(rows_, cols_);
            auto it = res.begin();
            for(const auto &v: *this)
                *it++ = v * num;
            return res;
        }
        friend MLPMatrix operator*(const double num, const MLPMatrix &other){
            return other * num;
        }
        MLPMatrix &operator*=(const double num) noexcept {
            for(auto &v: *this)
                v *= num;
            return *this;
        }


        MLPMatrix operator/(const double num) const{
            MLPMatrix res(rows_, cols_);
            auto it = res.begin();
            for(const auto &v: *this)
                *it++ = v / num;
            return res;
        }
        friend MLPMatrix operator/(const double num, const MLPMatrix &other){
            MLPMatrix res(other.rows_, other.cols_);
            auto it = res.begin();
            for(const auto &v: other)
                *it++ = num / v;
            return res;
        }
        MLPMatrix &operator/=(const double num) noexcept{
            for(auto&v : *this)
                v /= num;
            return *this;
        }


        MLPMatrix Exp() const { return Transform(std::exp); }
        double Sum() const noexcept{ return std::accumulate(begin(), end(), 0.0); }
        MLPMatrix Abs() const { return Transform(std::fabs); }


        MLPMatrix Transform(double(*foo)(double)) const{
            MLPMatrix res(rows_, cols_);
            std::transform(begin(), end(), res.begin(), foo);
            return res;
        }


        size_t Rows() const noexcept { return rows_; }
        size_t Cols() const noexcept { return cols_; }
        size_t Size() const noexcept {return rows_ * cols_;}


        double * begin() noexcept{ return matrix_[0]; }
        double * end() noexcept{ return matrix_[0] + Size(); }
        const double * begin() const noexcept{ return matrix_[0]; }
        const double * end() const noexcept{ return matrix_[0] + Size(); }
        const double * cbegin() const noexcept{ return matrix_[0]; }
        const double * cend() const noexcept{ return matrix_[0] + Size(); }


        double &operator()(const size_t i, const size_t j) noexcept{return matrix_[i][j];}
        const double &operator()(const size_t i, const size_t j) const noexcept {return matrix_[i][j];}


        friend std::ostream &operator<<(std::ostream &out, const MLPMatrix &other){
            for (size_t i = 0; i < other.Rows(); i++) {
                for (size_t j = 0; j < other.Cols(); j++) out << other(i, j) << " ";
                out << std::endl;
            }
            return out;
        }
        friend std::istream &operator>>(std::istream &in, MLPMatrix &other){
            for(auto& v : other) in >> v;
            return in;
        }

    private:
        size_t rows_, cols_;
        double **matrix_;
    };
}  // namespace S21
#endif //MULTILAYERABOBATRON_UTILS_MLPMATRIX_H_