#include "s21_matrix_oop.h"

namespace s21 {
    S21Matrix::S21Matrix() noexcept: rows_(0), cols_(0), matrix_(nullptr) {}

    S21Matrix::S21Matrix(const size_t rows, const size_t cols)
            : rows_(rows), cols_(cols) {
        if(rows_ && cols_) {
            matrix_ = new double *[rows_];
            matrix_[0] = new double[rows_ * cols_]();
            for (size_t i = 1; i < rows_; i++) matrix_[i] = matrix_[0] + i * cols_;
        }else{
            matrix_ = nullptr;
        }
    }


    S21Matrix::S21Matrix(const size_t rows, const size_t cols, std::mt19937& generator, const double from, const double to)
            :rows_(rows), cols_(cols){
        std::uniform_real_distribution<double> dist(from, to);
        matrix_ = new double *[rows_];
        matrix_[0] = new double [rows_ * cols_];
        for(size_t j = 0; j < cols_; ++j)
            matrix_[0][j] = dist(generator);
        for (size_t i = 1; i < rows_; i++) {
            matrix_[i] = matrix_[0] + i * cols_;
            for(size_t j = 0; j < cols_; ++j)
                matrix_[i][j] = dist(generator);
        }
    }

    S21Matrix::S21Matrix(const S21Matrix &other)
            : S21Matrix(other.rows_, other.cols_) {
        if(matrix_)
            std::memcpy(matrix_[0], other.matrix_[0], sizeof(double) * cols_ * rows_);
    }

    S21Matrix::S21Matrix(S21Matrix &&other) noexcept
            : rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.matrix_ = nullptr;
    }

    S21Matrix &S21Matrix::operator=(const S21Matrix &other) {
        if (this == &other) return *this;
        S21Matrix tmp(other);
        *this = std::move(tmp);
        return *this;
    }

    S21Matrix &S21Matrix::operator=(S21Matrix &&other) noexcept {
        if(&other == this) return *this;
        if(matrix_)
            delete[] matrix_[0];
        delete[] matrix_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        matrix_ = other.matrix_;
        other.rows_ = 0;
        other.cols_ = 0;
        other.matrix_ = nullptr;
        return *this;
    }

    S21Matrix::~S21Matrix() {
        if(matrix_)
            delete[] matrix_[0];
        delete[] matrix_;
    }

    bool S21Matrix::EqMatrix(const S21Matrix &other) const noexcept {
        if (Shape() != other.Shape()) return false;
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++)
                if (std::fabs(matrix_[i][j] - other.matrix_[i][j]) > 1.e-7)
                    return false;
        return true;
    }

    bool S21Matrix::operator==(const S21Matrix &other) const noexcept {
        return EqMatrix(other);
    }

    bool S21Matrix::operator!=(const S21Matrix &other) const noexcept {
        return !EqMatrix(other);
    }

    void S21Matrix::SumMatrix(const S21Matrix &other) {
        if (Shape() != other.Shape())
            throw std::logic_error("Sum: Matrices must have the same shape");
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++)
                matrix_[i][j] += other.matrix_[i][j];
    }

    S21Matrix S21Matrix::operator+(const S21Matrix &other) const {
        auto foo = [](double x, double y){ return x + y; };
        return ForEach(other, foo);
    }

    S21Matrix &S21Matrix::operator+=(const S21Matrix &other) {
        SumMatrix(other);
        return *this;
    }

    void S21Matrix::SubMatrix(const S21Matrix &other) {
        if (Shape() != other.Shape())
            throw std::logic_error("Sub: Matrices must have the same size");
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++)
                matrix_[i][j] = matrix_[i][j] - other.matrix_[i][j];
    }

    S21Matrix S21Matrix::operator-(const S21Matrix &other) const {
        auto foo = [](double x, double y){ return x - y; };
        return ForEach(other, foo);
    }

    S21Matrix S21Matrix::operator-() const{
        auto foo = [](double x){ return -x; };
        return ForEach(foo);
    }

    S21Matrix &S21Matrix::operator-=(const S21Matrix &other) {
        SubMatrix(other);
        return *this;
    }

    void S21Matrix::MulMatrix(const S21Matrix &other) {
        if (cols_ != other.rows_)
            throw std::logic_error(
                    "Amount of columns of the first matrix must match the amount of "
                    "rows of the second matrix");
        S21Matrix res(rows_, other.cols_);
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < other.cols_; j++)
                for (size_t k = 0; k < other.rows_; k++)
                    res.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
        *this = std::move(res);
    }

    S21Matrix S21Matrix::operator*(const S21Matrix &other) const {
        S21Matrix res(*this);
        res.MulMatrix(other);
        return res;
    }

    S21Matrix &S21Matrix::operator*=(const S21Matrix &other) {
        MulMatrix(other);
        return *this;
    }

    void S21Matrix::AddNumber(const double num) noexcept
    {
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++) matrix_[i][j] += num;
    }


    S21Matrix S21Matrix::operator+(const double num) const{
        auto foo = [num](double x){ return x + num; };
        return ForEach(foo);

    }

    S21Matrix operator+(const double num, const S21Matrix &other) {
        return other + num;
    }

    S21Matrix &S21Matrix::operator+=(const double num) noexcept {
        AddNumber(num);
        return *this;
    }

    void S21Matrix::SubNumber(const double num) noexcept
    {
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++) matrix_[i][j] -= num;
    }


    S21Matrix S21Matrix::operator-(const double num) const{
        auto foo = [num](double x){ return x - num; };
        return ForEach(foo);
    }

    S21Matrix operator-(const double num, const S21Matrix &other) {
        auto foo = [num](double x){ return num - x; };
        return other.ForEach(foo);
    }

    S21Matrix &S21Matrix::operator-=(const double num) noexcept {
        SubNumber(num);
        return *this;
    }

    void S21Matrix::MulNumber(const double num) noexcept {
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++) matrix_[i][j] *= num;
    }

    S21Matrix S21Matrix::operator*(const double num) const {
        auto foo = [num](double x){ return x * num; };
        return ForEach(foo);
    }

    S21Matrix operator*(const double &x, const S21Matrix &other) {
        return other * x;
    }

    S21Matrix &S21Matrix::operator*=(const double num) noexcept {
        MulNumber(num);
        return *this;
    }

    void S21Matrix::DivNumber(const double num) noexcept {
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++) matrix_[i][j] /= num;
    }

    S21Matrix S21Matrix::operator/(const double num) const {
        auto fun = [num](double x){ return x / num; };
        return ForEach(fun);

    }

    S21Matrix operator/(const double &x, const S21Matrix &other) {
        auto fun = [x](double y){ return x / y; };
        return other.ForEach(fun);
    }

    S21Matrix &S21Matrix::operator/=(const double num) noexcept {
        DivNumber(num);
        return *this;
    }

    S21Matrix S21Matrix::MulElementwise(const S21Matrix& other) const{
        if(Shape() != other.Shape())
            throw std::logic_error("Matrices must have the same size");
        S21Matrix res(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                res.matrix_[i][j] = other.matrix_[i][j] * matrix_[i][j];
        return res;
    }


    S21Matrix S21Matrix::MulByTransposed(const S21Matrix &other) const{
        if (cols_ != other.cols_)
            throw std::logic_error(
                    "MulByT: Amount of columns of the first matrix must match the amount of "
                    "columns of the second matrix");
        S21Matrix res(rows_, other.rows_);
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < other.rows_; j++)
                for (size_t k = 0; k < cols_; k++)
                    res.matrix_[i][j] += matrix_[i][k] * other.matrix_[j][k];
        return res;
    }

    S21Matrix S21Matrix::MulSelfTranspose(const S21Matrix &other) const {
        if (rows_ != other.rows_)
            throw std::logic_error("MulSelfT: Amount of rows of the first matrix must match the amount of "
                                   "rows of the second matrix");
        S21Matrix res(cols_, other.cols_);
        for (size_t i = 0; i < cols_; i++)
            for (size_t j = 0; j < other.cols_; j++)
                for (size_t k = 0; k < rows_; k++)
                    res.matrix_[i][j] += matrix_[k][i] * other.matrix_[k][j];
        return res;
    }
    S21Matrix S21Matrix::Transpose() const {
        if (matrix_ == nullptr) throw std::invalid_argument("T: Matrix is NULL");
        S21Matrix res(cols_, rows_);
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < cols_; j++) res.matrix_[j][i] = matrix_[i][j];
        return res;
    }

    S21Matrix S21Matrix::T() const{
        return Transpose();
    }

    S21Matrix S21Matrix::Exp() const{
        auto foo = [](double x){ return std::exp(x); };
        return ForEach(foo);
    }

    double S21Matrix::Sum() const noexcept{
        double res(0.0);
        for(size_t i = 0; i < rows_; ++i)
            for(size_t j = 0; j < cols_; ++j)
                res += matrix_[i][j];
        return res;
    }

    S21Matrix S21Matrix::Abs() const {
        auto fun = [](double x){ return std::fabs(x); };
        return ForEach(fun);
    }


    S21Matrix S21Matrix::ForEach(const std::function<double(const double)> &function) const {
        S21Matrix res(rows_, cols_);
        for(size_t i = 0; i < rows_; ++i)
            for(size_t j = 0; j< cols_; ++j)
                res.matrix_[i][j] = function(matrix_[i][j]);
        return res;
    }

    S21Matrix S21Matrix::ForEach(const S21Matrix& other, const std::function<double(const double, const double)> &function) const{
        if(Shape() != other.Shape())
            throw std::logic_error("ForEach: Matrices must have the same shape");
        S21Matrix res(rows_, cols_);
        for(size_t i = 0; i < rows_; ++i)
            for(size_t j = 0; j < cols_; ++j)
                res.matrix_[i][j] = function(matrix_[i][j], other.matrix_[i][j]);
        return res;
    }

    void S21Matrix::SetCols(const size_t cols) {
        SetShape(rows_, cols);
    }

    void S21Matrix::SetRows(const size_t rows) {
        SetShape(rows, cols_);
    }

    void S21Matrix::SetShape(const size_t rows, const size_t cols) {
        if(rows_ == rows && cols_ == cols) return;
        if(!rows_ || ! cols_) throw std::length_error("Invalid Matrix size");
        S21Matrix tmp(rows, cols);
        for (size_t i = 0; i < rows_ && i < rows; i++)
            std::memcpy(
                    tmp.matrix_[i], matrix_[i],
                    cols_ < cols ? cols_ * sizeof(double) : cols * sizeof(double));
        *this = std::move(tmp);
    }

    std::ostream &operator<<(std::ostream &out, const S21Matrix &other) {
        for (int i = 0; i < other.Rows(); i++) {
            for (int j = 0; j < other.Cols(); j++) out << other(i, j) << " ";
            out << std::endl;
        }
        return out;
    }

    std::istream &operator>>(std::istream &in, S21Matrix &other) {
        for (int i = 0; i < other.Rows(); i++) {
            for (int j = 0; j < other.Cols(); j++) in >> other(i, j);
        }
        return in;
    }

}  // namespace S21