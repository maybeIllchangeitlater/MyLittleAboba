#ifndef MULTILAYERABOBATRON_UTILS_MATRIX_H_
#define MULTILAYERABOBATRON_UTILS_MATRIX_H_

#include <cmath>
#include <cstring>  //for memcpy
#include <iostream>
#include <random>
#include <functional>

namespace s21 {
    class Matrix {
    public:

        Matrix() noexcept: rows_(0), cols_(0), matrix_(nullptr){}

        explicit Matrix(const size_t rows, const size_t cols)  : rows_(rows), cols_(cols) {
            if(rows_ && cols_) {
                matrix_ = new double *[rows_];
                matrix_[0] = new double[rows_ * cols_]();
                for (size_t i = 1; i < rows_; i++) matrix_[i] = matrix_[0] + i * cols_;
            }else{
                matrix_ = nullptr;
            }
        }
        /**
         * @brief constructs a matrix filled with random values in range(from, to)\n
         * using generator of type std::mt19937
         */
        Matrix(const size_t rows, const size_t cols, std::mt19937& generator, const double from, const double to)
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
        /**
         * @brief Constructs vector-like matrix(1 by x where x is iterable object size)\n
         * Object must be iterable, have .size() method\n
         * and have iterator dereference value implicitly convertable to double\n
         */
        template<typename Iterable>
        Matrix(const Iterable& other)
        : Matrix(1, other.size()) {
            std::copy(other.begin(), other.end(), begin());
        }
        Matrix(const Matrix &other): Matrix(other.rows_, other.cols_) {
            if(matrix_)
                std::memcpy(matrix_[0], other.matrix_[0], sizeof(double) * cols_ * rows_);
        }
        Matrix(Matrix &&other) noexcept: rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
            other.rows_ = 0;
            other.cols_ = 0;
            other.matrix_ = nullptr;
        }
        Matrix &operator=(const Matrix &other){
            if (this == &other) return *this;
            Matrix tmp(other);
            *this = std::move(tmp);
            return *this;
        }
        Matrix &operator=(Matrix &&other) noexcept{
            if(&other == this) return *this;
            if(matrix_) delete[] matrix_[0];
            delete[] matrix_;
            rows_ = other.rows_, cols_ = other.cols_, matrix_ = other.matrix_;
            other.rows_ = 0, other.cols_ = 0;
            other.matrix_ = nullptr;
            return *this;
        }
        template<typename Iterable>
        Matrix& operator=(const Iterable& other){
            Matrix tmp(other);
            *this = std::move(tmp);
            return *this;
        }
        ~Matrix(){
            if(matrix_) delete[] matrix_[0];
            delete[] matrix_;
        }

        template<typename Iterable>
        Matrix operator+(const Iterable &other) const{
            Matrix res(rows_, cols_);
            std::transform(begin(), end(), other.begin(), res.begin(), std::plus<>());
            return res;
        }
        template<typename Iterable>
        Matrix &operator+=(const Iterable &other){
            std::transform(begin(), end(), other.begin(), begin(), std::plus<>());
            return *this;
        }


        template<typename Iterable>
        Matrix operator-(const Iterable &other) const{
            Matrix res(rows_, cols_);
            std::transform(begin(), end(), other.begin(), res.begin(), std::minus<>());
            return res;
        }
        template<typename Iterable>
        Matrix &operator-=(const Iterable &other){
            std::transform(begin(), end(), other.begin(), begin(), std::minus<>());
            return *this;
        }

        ///hadamard product
        template<typename Iterable>
        Matrix operator&(const Iterable& other) const{
            Matrix res(rows_, cols_);
            std::transform(begin(), end(), other.begin(), res.begin(), std::multiplies<>());
            return res;
        }
        template<typename Iterable>
        Matrix &operator&=(const Iterable& other) {
            std::transform(begin(), end(), other.begin(), begin(), std::multiplies<>());
            return *this;
        }


        Matrix operator*(const Matrix &other) const{
            Matrix res(rows_, other.cols_);
            for (size_t i = 0; i < rows_; i++)
                for (size_t j = 0; j < other.cols_; j++)
                    for (size_t k = 0; k < other.rows_; k++)
                        res.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
            return res;
        }
        Matrix &operator*=(const Matrix &other){
            *this = (*this * other);
            return *this;
        }
        Matrix MulByTransposed(const Matrix& other) const{
            Matrix res(rows_, other.rows_);
            for (size_t i = 0; i < rows_; i++)
                for (size_t j = 0; j < other.rows_; j++)
                    for (size_t k = 0; k < cols_; k++)
                        res.matrix_[i][j] += matrix_[i][k] * other.matrix_[j][k];
            return res;
        }
        Matrix MulSelfTranspose(const Matrix &other) const{
            Matrix res(cols_, other.cols_);
            for (size_t i = 0; i < cols_; i++)
                for (size_t j = 0; j < other.cols_; j++)
                    for (size_t k = 0; k < rows_; k++)
                        res.matrix_[i][j] += matrix_[k][i] * other.matrix_[k][j];
            return res;
        }

        Matrix operator*(const double num) const{
            Matrix res(rows_, cols_);
            auto it = res.begin();
            for(const auto &v: *this)
                *it++ = v * num;
            return res;
        }
        friend Matrix operator*(const double num, const Matrix &other){
            return other * num;
        }
        Matrix &operator*=(const double num) noexcept {
            for(auto &v: *this)
                v *= num;
            return *this;
        }


        Matrix operator/(const double num) const{
            Matrix res(rows_, cols_);
            auto it = res.begin();
            for(const auto &v: *this)
                *it++ = v / num;
            return res;
        }
        friend Matrix operator/(const double num, const Matrix &other){
            Matrix res(other.rows_, other.cols_);
            auto it = res.begin();
            for(const auto &v: other)
                *it++ = num / v;
            return res;
        }
        Matrix &operator/=(const double num) noexcept{
            for(auto&v : *this)
                v /= num;
            return *this;
        }


        Matrix Exp() const { return Transform(std::exp); }
        double Sum() const noexcept{ return std::accumulate(begin(), end(), 0.0); }
        Matrix Abs() const { return Transform(std::fabs); }
        Matrix Pow2() const { return Transform([](double x){ return x * x; }); }


        Matrix Transform(double(*foo)(double)) const{
            Matrix res(rows_, cols_);
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


        friend std::ostream &operator<<(std::ostream &out, const Matrix &other){
            for (size_t i = 0; i < other.Rows(); i++) {
                for (size_t j = 0; j < other.Cols(); j++) out << other(i, j) << " ";
                out << std::endl;
            }
            return out;
        }
        friend std::istream &operator>>(std::istream &in, Matrix &other){
            for(auto& v : other) in >> v;
            return in;
        }

    private:
        size_t rows_, cols_;
        double **matrix_;
    };
}  // namespace S21
#endif //MULTILAYERABOBATRON_UTILS_MATRIX_H_