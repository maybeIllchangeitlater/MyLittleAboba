#ifndef CPP1_S21_MATRIXPLUS_7_S21_MATRIX_OOP_H
#define CPP1_S21_MATRIXPLUS_7_S21_MATRIX_OOP_H

#include <cmath>
#include <cstring>  //for memcpy & memset
#include <iostream>
#include <random>
#include <iterator>
#include <functional>
#include <algorithm>

namespace s21 {
class S21Matrix {
 public:
    using iterator = double*;
    using const_iterator = const double*;
    using value_type = double;
    enum InitMode{
        kZero
    };
  S21Matrix() noexcept : rows_(0), cols_(0), size_(0), matrix_(nullptr) {}
  explicit S21Matrix(const size_t rows, const size_t cols) : rows_(rows), cols_(cols), size_(rows_ * cols_) {
      matrix_ = static_cast<double *>(::operator new[](size_ * sizeof(double)));
  }
  explicit S21Matrix(const size_t rows, const size_t cols, InitMode) : S21Matrix(rows, cols) {
      std::memset(matrix_, 0, size_ * sizeof(double));
  }
  explicit S21Matrix(const std::pair<size_t, size_t> shape) : S21Matrix(shape.first, shape.second) {}
  explicit S21Matrix(const std::pair<size_t, size_t> shape, InitMode im) : S21Matrix(shape.first, shape.second, im) {}

  S21Matrix(const size_t rows, const size_t cols, std::mt19937& generator, const double from, const double to)
  : S21Matrix(rows, cols) {
      std::uniform_real_distribution<double> dist(from, to);
      for (size_t i = 0; i < size_; i++)
              matrix_[i] = dist(generator);

  }
  S21Matrix(const S21Matrix &other) : S21Matrix(other.rows_, other.cols_) {
      std::memcpy(matrix_, other.matrix_, size_ * sizeof(double));
  }
  S21Matrix(S21Matrix &&other) noexcept
  : rows_(other.rows_), cols_(other.cols_), size_(other.size_), matrix_(other.matrix_){
      other.rows_ = 0;
      other.cols_ = 0;
      other.size_ = 0;
      other.matrix_ = nullptr;
  }
  S21Matrix &operator=(const S21Matrix &other){
      if (this == &other)
          return *this;
      S21Matrix tmp(other);
      *this = std::move(tmp);
      return *this;
  }
  S21Matrix &operator=(S21Matrix &&other) noexcept{
      ::operator delete[](matrix_);
      rows_ = other.rows_;
      cols_ = other.cols_;
      size_ = other.size_;
      matrix_ = other.matrix_;
      other.rows_ = 0;
      other.cols_ = 0;
      other.size_ = 0;
      other.matrix_ = nullptr;
      return *this;
  };
  ~S21Matrix(){
    ::operator delete[](matrix_);
  }

  bool EqMatrix(const S21Matrix &other) const noexcept {
      if (Shape() != other.Shape())
          return false;
      for (size_t i = 0; i < size_; ++i)
          if(std::fabs(matrix_[i] - other.matrix_[i]) > 1e-7)
              return false;
      return true;
  }
  bool operator==(const S21Matrix &other) const noexcept { return EqMatrix(other); }
  bool operator!=(const S21Matrix &other) const noexcept { return !EqMatrix(other); }

  void SumMatrix(const S21Matrix &other) {
      if(Shape() != other.Shape())
          throw std::logic_error("Sum: Matrices must have the same shape");
      for(size_t i = 0; i < size_; ++i)
          matrix_[i] += other.matrix_[i];
  }
  S21Matrix operator+(const S21Matrix &other) const {
      S21Matrix res(*this);
      res.SumMatrix(other);
      return res;
  }
  S21Matrix &operator+=(const S21Matrix &other) {
      SumMatrix(other);
      return *this;
  }

  void SubMatrix(const S21Matrix &other) {
      if(Shape() != other.Shape())
          throw std::logic_error("Sub: Matrices must have the same shape");
      for(size_t i = 0; i < size_; ++i)
          matrix_[i] -= other.matrix_[i];
  }
  S21Matrix operator-(const S21Matrix &other) const {
      S21Matrix res(*this);
      res.SubMatrix(other);
      return res;
  }
  S21Matrix &operator-=(const S21Matrix &other) {
      SubMatrix(other);
      return *this;
  }

  S21Matrix Transpose() const {
      S21Matrix res(cols_, rows_);
      for (size_t i = 0; i < rows_; i++)
          for (size_t j = 0; j < cols_; j++)
              res(j, i) = (*this)(i, j);
      return res;
  }
  S21Matrix T() const { return Transpose(); }

  S21Matrix MulElementwise(const S21Matrix& other) const {
      if (Shape() != other.Shape())
          throw std::logic_error("Mul elementwise: Matrices must have the same shape");
      S21Matrix res(rows_, cols_);
      for(size_t i = 0; i < size_; ++i)
          res[i] = matrix_[i] * other.matrix_[i];
      return res;
  }

  void MulMatrix(const S21Matrix &other) {
      if (cols_ != other.rows_)
          throw std::logic_error(
                  "Mul: Amount of columns of the first matrix must match the amount of "
                  "rows of the second matrix");
      S21Matrix res(rows_, other.cols_, kZero);
      for (size_t i = 0; i < rows_; i++)
          for (size_t j = 0; j < other.cols_; j++)
              for (size_t k = 0; k < other.rows_; k++)
                  res(i, j) += (*this)(i, k) * other(k , j);
      *this = std::move(res);
  }
  S21Matrix MulByTransposed(const S21Matrix& other) const {
      if (cols_ != other.cols_)
          throw std::logic_error(
                  "MulByT: Amount of columns of the first matrix must match the amount of "
                  "columns of the second matrix");
      S21Matrix res(rows_, other.rows_, kZero);
      for (size_t i = 0; i < rows_; i++)
          for (size_t j = 0; j < other.rows_; j++)
              for (size_t k = 0; k < cols_; k++)
                  res(i, j) += (*this)(i, k) * other(j, k);
      return res;
  }
  S21Matrix MulSelfTranspose(const S21Matrix &other) const {
      if (rows_ != other.rows_)
          throw std::logic_error("MulSelfT: Amount of rows of the first matrix must match the amount of "
                                 "rows of the second matrix");
      S21Matrix res(cols_, other.cols_, kZero);
      for (size_t i = 0; i < cols_; i++)
          for (size_t j = 0; j < other.cols_; j++)
              for (size_t k = 0; k < rows_; k++)
                  res(i, j) += (*this)(k, i) * other(k, j);
      return res;
  }
  S21Matrix operator*(const S21Matrix &other) const {
      S21Matrix res(*this);
      res.MulMatrix(other);
      return res;
  }
  S21Matrix &operator*=(const S21Matrix &other) {
      MulMatrix(other);
      return *this;
  }


  S21Matrix ForEach(const std::function<double(double)> &function) const {
      S21Matrix res(*this);
      std::for_each(res.begin(), res.end(), function);
      return res;
  }

  double Sum() const noexcept{ return std::accumulate(begin(), end(), 0.0); }
  S21Matrix Exp() const { return ForEach([](double x){ return std::exp(x); }); }
  S21Matrix Abs() const { return ForEach([](double x){ return std::fabs(x); }); }

  size_t Rows() const noexcept { return rows_; }
  size_t Cols() const noexcept { return cols_; }
  std::pair<size_t, size_t> Shape() const noexcept { return std::make_pair(rows_, cols_); }
  size_t Size() const noexcept { return size_; }

  void ResizeRows(const size_t rows) { Resize(rows, cols_); }
  void ResizeCols(const size_t cols) { Resize(rows_, cols); }
  void Resize(const size_t rows, const size_t cols) {
      if(rows_ == rows && cols_ == cols)
          return;
      S21Matrix tmp(rows, cols, kZero);
      std::memcpy(tmp.matrix_, matrix_, std::min(size_, tmp.size_) * sizeof(double));
      *this = std::move(tmp);
  }

  double &at(const size_t i, const size_t j) {
      if(i >= rows_ || j >= cols_)
          throw std::out_of_range("Access: Out of Matrix bounds");
      return matrix_[i * j + j];
  }
  const double &at(const size_t i, const size_t j) const {
        if(i >= rows_ || j >= cols_)
            throw std::out_of_range("Access: Out of Matrix bounds");
        return matrix_[i * j + j];
  }
  double &operator()(const size_t i, const size_t j) noexcept { return matrix_[i * j + j]; }
  const double &operator()(const size_t i, const size_t j) const noexcept { return matrix_[i * j + j]; }
  double &operator[](const size_t i) noexcept { return matrix_[i]; }
  const double &operator[](const size_t i) const noexcept { return matrix_[i]; }

  iterator begin() noexcept { return matrix_; }
  const_iterator begin() const noexcept { return matrix_; }
  iterator end() noexcept{ return matrix_ + rows_ * cols_; }
  const_iterator end() const noexcept { return matrix_ + rows_ * cols_; }
  const_iterator cbegin() const noexcept { return matrix_; }
  const_iterator cend() const noexcept{ return matrix_ + rows_ * cols_; }

  /**********************************************Fluff**************************************************/
  S21Matrix operator-() const {
      S21Matrix res(rows_, cols_);
      for (size_t i = 0; i < size_; ++i)
          res.matrix_[i] = -matrix_[i];
      return res;
  }

    void AddNumber(const double num) noexcept {
      std::for_each(begin(), end(), [num](double x){ return x + num; });
    }


    S21Matrix operator+(const double num) const{
        S21Matrix res(*this);
        res.AddNumber(num);
        return res;
    }

    friend S21Matrix operator+(const double num, const S21Matrix &other) {
        S21Matrix res(other);
        res.AddNumber(num);
        return res;
    }

    S21Matrix &operator+=(const double num) noexcept {
        AddNumber(num);
        return *this;
    }

    void SubNumber(const double num) noexcept {
        std::for_each(begin(), end(), [num](double x){ return x - num; });
    }


    S21Matrix operator-(const double num) const{
        S21Matrix res(*this);
        res.SubNumber(num);
        return res;
    }

    friend S21Matrix operator-(const double num, const S21Matrix &other) {
        S21Matrix res(other.rows_, other.cols_);
        for (size_t i = 0; i < other.Size(); ++i)
            res[i] = num - other[i];
        return res;
    }

    S21Matrix & operator-=(const double num) noexcept {
        SubNumber(num);
        return *this;
    }

    void MulNumber(const double num) noexcept {
        std::for_each(begin(), end(), [num](double x){ return x * num; });
    }

    S21Matrix operator*(const double &x) const {
        S21Matrix res(*this);
        res.MulNumber(x);
        return res;
    }

    friend S21Matrix operator*(const double &x, const S21Matrix &other) {
        S21Matrix res(other);
        res.MulNumber(x);
        return res;
    }

    S21Matrix &operator*=(const double &x) noexcept {
        MulNumber(x);
        return *this;
    }

    void DivNumber(const double num) noexcept {
        std::for_each(begin(), end(), [num](double x){ return x / num; });
    }

    S21Matrix operator/(const double &x) const {
        S21Matrix res(*this);
        res.DivNumber(x);
        return res;
    }

    friend S21Matrix operator/(const double &x, const S21Matrix &other) {
        S21Matrix res(other.rows_, other.cols_);
        for (size_t i = 0; i < other.Size(); i++)
            res.matrix_[i] = x / other.matrix_[i];
        return res;
    }

    S21Matrix &operator/=(const double &x) noexcept {
        DivNumber(x);
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &out, const S21Matrix &m) {
      for (size_t i = 0; i < m.Rows(); i++) {
          for (size_t j = 0; j < m.Cols(); j++) {
              out << m(i, j) << " ";
          }
          out << std::endl;
      }
      return out;
  }
  friend std::istream &operator>>(std::istream &in, S21Matrix &m) {
      for (size_t i = 0; i < m.Rows(); i++)
          for (size_t j = 0; j < m.Cols(); j++)
              in >> m(i, j);
      return in;
  }

 private:
  size_t rows_, cols_, size_;
  double *matrix_;
};
}  // namespace S21
#endif
