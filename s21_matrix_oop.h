#ifndef CPP1_S21_MATRIXPLUS_7_S21_MATRIX_OOP_H
#define CPP1_S21_MATRIXPLUS_7_S21_MATRIX_OOP_H

#include <cmath>
#include <cstring>  //for memcpy
#include <iostream>
#include <random>
#include <functional>
//#include <cstdlib> // for srand

namespace s21 {
class S21Matrix {
 public:

  S21Matrix() noexcept;
  explicit S21Matrix(const size_t rows, const size_t cols);
  S21Matrix(const size_t rows, const size_t cols, bool zeroes);
  S21Matrix(const size_t rows, const size_t cols, std::mt19937& generator, const double from, const double to);
  S21Matrix(const S21Matrix &other);
  S21Matrix(S21Matrix &&other) noexcept;
  S21Matrix &operator=(const S21Matrix &other);
  S21Matrix &operator=(S21Matrix &&other) noexcept;
  ~S21Matrix();

  bool EqMatrix(const S21Matrix &other) const noexcept;
  bool operator==(const S21Matrix &other) const noexcept;
  bool operator!=(const S21Matrix &other) const noexcept;

  void SumMatrix(const S21Matrix &other);
  S21Matrix operator+(const S21Matrix &other) const;
  S21Matrix &operator+=(const S21Matrix &other);

  void SubMatrix(const S21Matrix &other);
  S21Matrix operator-(const S21Matrix &other) const;
  S21Matrix operator-() const;
  S21Matrix &operator-=(const S21Matrix &other);

  void MulMatrix(const S21Matrix &other);
  S21Matrix operator*(const S21Matrix &other) const;
  S21Matrix &operator*=(const S21Matrix &other);

  void AddNumber(const double num) noexcept;
  S21Matrix operator+(const double num) const;
  friend S21Matrix operator+(const double num, const S21Matrix &other);
  S21Matrix &operator+=(const double num) noexcept;

  void SubNumber(const double num) noexcept;
  S21Matrix operator-(const double num) const;
  friend S21Matrix operator-(const double num, const S21Matrix &other);
  S21Matrix &operator-=(const double num) noexcept;

  void MulNumber(const double num) noexcept;
  S21Matrix operator*(const double num) const;
  friend S21Matrix operator*(const double num, const S21Matrix &other);
  S21Matrix &operator*=(const double num) noexcept;

  void DivNumber(const double num) noexcept;
  S21Matrix operator/(const double num) const;
  friend S21Matrix operator/(const double num, const S21Matrix &other);
  S21Matrix &operator/=(const double num) noexcept;

  S21Matrix MulElementwise(const S21Matrix& other) const;
  S21Matrix MulByTransposed(const S21Matrix& other) const;
  S21Matrix MulSelfTranspose(const S21Matrix &other) const;

  S21Matrix Transpose() const;
  S21Matrix T() const;
  S21Matrix Exp() const;
  double Sum() const noexcept;
  S21Matrix Abs() const;

  S21Matrix ForEach(const std::function<double(const double)> &function) const;
  S21Matrix ForEach(const S21Matrix& other, const std::function<double(const double, const double)> &function) const;


  size_t Rows() const noexcept { return rows_; }
  size_t Cols() const noexcept { return cols_; }
  std::pair<size_t, size_t> Shape() const noexcept {return std::make_pair(rows_, cols_);}
  size_t Size() const noexcept {return rows_ * cols_;}

  void SetRows(const size_t rows);
  void SetCols(const size_t cols);
  void SetShape(const size_t rows, const size_t cols);

  double &operator()(const size_t i, const size_t j) noexcept{return matrix_[i][j];}
  const double &operator()(const size_t i, const size_t j) const noexcept {return matrix_[i][j];}


  friend std::ostream &operator<<(std::ostream &out, const S21Matrix &other);
  friend std::istream &operator>>(std::istream &in, S21Matrix &other);

 private:
  size_t rows_, cols_;
  double **matrix_;
};
}  // namespace S21
#endif
