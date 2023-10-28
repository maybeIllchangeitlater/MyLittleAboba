#include "s21_matrix_oop.h"

namespace s21 {
S21Matrix::S21Matrix() noexcept: rows_(0), cols_(0), matrix_(nullptr) {}

S21Matrix::S21Matrix(const size_t rows, const size_t cols)
    : rows_(rows), cols_(cols) {
  matrix_ = new double *[rows_]();
  for (size_t i = 0; i < rows_; i++) matrix_[i] = new double[cols_]();
}

S21Matrix::S21Matrix(const size_t rows, const size_t cols, std::mt19937& generator, const double from, const double to)
:rows_(rows), cols_(cols){
    std::uniform_real_distribution<double> dist(from, to);
    matrix_ = new double *[rows_];
    for (size_t i = 0; i < rows_; i++) {
        matrix_[i] = new double[cols_];
        for(size_t j = 0; j < cols_; ++j){
            matrix_[i][j] = dist(generator);
        }
    }
}

S21Matrix::S21Matrix(const S21Matrix &other)
    : rows_(other.rows_), cols_(other.cols_) {
  if (other.matrix_ == nullptr)
    matrix_ = nullptr;
  else {
    matrix_ = new double *[rows_]();
    for (size_t i = 0; i < rows_; i++) {
      matrix_[i] = new double[cols_]();
      std::memcpy(matrix_[i], other.matrix_[i], sizeof(double) * cols_);
    }
  }
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
  for (size_t i = 0; i < rows_; i++) delete[] matrix_[i];
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
  for (size_t i = 0; i < rows_; i++) delete[] matrix_[i];
  delete[] matrix_;
}

bool S21Matrix::EqMatrix(const S21Matrix &other) const noexcept {
  if (rows_ != other.rows_ || cols_ != other.cols_) return false;
  bool res = true;
  for (size_t i = 0; i < rows_; i++)
    for (size_t j = 0; j < cols_; j++)
      if (std::fabs(matrix_[i][j] - other.matrix_[i][j]) > 1.e-7)
        res = false;
  return res;
}

bool S21Matrix::operator==(const S21Matrix &other) const noexcept {
  return EqMatrix(other);
}

bool S21Matrix::operator!=(const S21Matrix &other) const noexcept {
  return !EqMatrix(other);
}

void S21Matrix::SumMatrix(const S21Matrix &other) {
    if (GetShape() != other.GetShape())
        throw std::logic_error("Matrices must have the same size");
  for (size_t i = 0; i < rows_; i++)
    for (size_t j = 0; j < cols_; j++)
      matrix_[i][j] = matrix_[i][j] + other.matrix_[i][j];
}

S21Matrix S21Matrix::SumColwise(const S21Matrix & other) const{
    if(rows_ != other.rows_)
        throw std::logic_error("Amount of rows didnt match");
    S21Matrix res(rows_, cols_);
    for(size_t i = 0; i < rows_; ++i)
        for(size_t j = 0; j < cols_; ++j)
//            for(size_t k = 0; k < other.cols_; ++k) logical but not needed for mlp and slower
            res.matrix_[i][j] += other.matrix_[i][0];
    return res;
}

S21Matrix S21Matrix::operator+(const S21Matrix &other) const {
  S21Matrix res(*this);
  res.SumMatrix(other);
  return res;
}

S21Matrix &S21Matrix::operator+=(const S21Matrix &other) {
  SumMatrix(other);
  return *this;
}

void S21Matrix::SubMatrix(const S21Matrix &other) {
  if (GetShape() != other.GetShape())
    throw std::logic_error("Matrices must have the same size");
  for (size_t i = 0; i < rows_; i++)
    for (size_t j = 0; j < cols_; j++)
      matrix_[i][j] = matrix_[i][j] - other.matrix_[i][j];
}

S21Matrix S21Matrix::operator-(const S21Matrix &other) const {
  S21Matrix res(*this);
  res.SubMatrix(other);
  return res;
}

S21Matrix S21Matrix::operator-() const{
    S21Matrix res(rows_, cols_);
    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++)
            res.matrix_[i][j] = -matrix_[i][j];
    return res;
}

S21Matrix &S21Matrix::operator-=(const S21Matrix &other) {
  SubMatrix(other);
  return *this;
}

void S21Matrix::MulMatrix(const S21Matrix &other) {
  if (cols_ != other.rows_)
    throw std::logic_error(
        "Amount of columns of the first matrix must be equal to amount of "
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
    S21Matrix res(*this);
    res.AddNumber(num);
    return res;
}

S21Matrix operator+(const double num, const S21Matrix &other) {
  S21Matrix res(other);
  res.AddNumber(num);
  return res;
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
    S21Matrix res(*this);
    res.SubNumber(num);
    return res;
}

S21Matrix operator-(const double num, const S21Matrix &other) {
    S21Matrix res(other.rows_, other.cols_);
    for (size_t i = 0; i < other.rows_; i++)
        for (size_t j = 0; j < other.cols_; j++) other.matrix_[i][j] = num - other.matrix_[i][j];
    return res;
}

S21Matrix &S21Matrix::operator-=(const double num) noexcept {
    SubNumber(num);
    return *this;
}

void S21Matrix::MulNumber(const double num) noexcept {
  for (size_t i = 0; i < rows_; i++)
    for (size_t j = 0; j < cols_; j++) matrix_[i][j] *= num;
}

S21Matrix S21Matrix::operator*(const double &x) const {
  S21Matrix res(*this);
  res.MulNumber(x);
  return res;
}

S21Matrix operator*(const double &x, const S21Matrix &other) {
  S21Matrix res(other);
  res.MulNumber(x);
  return res;
}

S21Matrix &S21Matrix::operator*=(const double &x) noexcept {
  MulNumber(x);
  return *this;
}

void S21Matrix::DivNumber(const double num) noexcept {
    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++) matrix_[i][j] /= num;
}

S21Matrix S21Matrix::operator/(const double &x) const {
    S21Matrix res(*this);
    res.DivNumber(x);
    return res;
}

S21Matrix operator/(const double &x, const S21Matrix &other) {
    S21Matrix res(other);
    for (size_t i = 0; i < res.rows_; i++)
        for (size_t j = 0; j < res.cols_; j++) res.matrix_[i][j] = x / res.matrix_[i][j];
    return res;
}

S21Matrix &S21Matrix::operator/=(const double &x) noexcept {
    DivNumber(x);
    return *this;
}

S21Matrix S21Matrix::MulElementwise(const S21Matrix& other) const{
    if(GetShape() != other.GetShape())
        throw std::logic_error("Matrices must have the same size");
    S21Matrix res(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            res.matrix_[i][j] = other.matrix_[i][j] * matrix_[i][j];
    return res;
}

S21Matrix S21Matrix::MulElementwiseT(const S21Matrix& other) const{
    if(rows_ != other.cols_ || cols_ != other.rows_)
        throw std::logic_error("Matrices must have the same size after transpose");
    S21Matrix res(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            res.matrix_[i][j] = other.matrix_[j][i] * matrix_[i][j];
    return res;
}

S21Matrix S21Matrix::MulByTransposed(const S21Matrix &other) const{
        if (cols_ != other.cols_)
            throw std::logic_error(
                    "Amount of columns of the first matrix must be equal to amount of "
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
        throw std::logic_error("Amount of rows of the first matrix must be equal to amount of "
                    "rows of the second matrix");
    S21Matrix res(cols_, other.cols_);
    for (size_t i = 0; i < cols_; i++)
        for (size_t j = 0; j < other.cols_; j++)
            for (size_t k = 0; k < rows_; k++)
                res.matrix_[i][j] += matrix_[k][i] * other.matrix_[k][j];
    return res;
}
S21Matrix S21Matrix::Transpose() const {
  if (matrix_ == nullptr) throw std::invalid_argument("Matrix is NULL");
  S21Matrix res(cols_, rows_);
  for (size_t i = 0; i < rows_; i++)
    for (size_t j = 0; j < cols_; j++) res.matrix_[j][i] = matrix_[i][j];
  return res;
}

S21Matrix S21Matrix::T() const{
    return Transpose();
}

S21Matrix S21Matrix::Exp() const{
    S21Matrix res(rows_, cols_);
    for(size_t i = 0; i < rows_; ++i)
        for(size_t j = 0; j < cols_; ++j)
            res.matrix_[i][j] = std::exp(matrix_[i][j]);
    return res;
}

double S21Matrix::Sum() const noexcept{
    double res(0.0);
    for(size_t i = 0; i < rows_; ++i)
        for(size_t j = 0; j < cols_; ++j)
            res += matrix_[i][j];
    return res;
}

double S21Matrix::Determinant() const {
  if (rows_ != cols_) throw std::logic_error("The matrix is not square");
  if (rows_ == 1) return matrix_[0][0];
  int current_start = 0;
  double first = 0.0;
  double det = 1.0;
  S21Matrix tmp(*this);
  for (; current_start < tmp.rows_ && det != 0.0; current_start++) {
    size_t i = current_start;
    if (fabs(tmp.matrix_[current_start][current_start]) < 1e-7) {
      for (; i < tmp.rows_ && fabs(tmp.matrix_[i][current_start]) < 1e-7; i++)
        ;
      i == tmp.rows_ ? det = 0.0 : det *= -1,
                       std::swap(tmp.matrix_[current_start], tmp.matrix_[i]);
    }
    det *= tmp.matrix_[current_start][current_start];
    for (size_t row = current_start; row < tmp.rows_; row++) {
      first = tmp.matrix_[row][current_start];
      for (size_t col = current_start; col < tmp.cols_; col++) {
        row == current_start
            ? tmp.matrix_[row][col] /= first
            : tmp.matrix_[row][col] -= tmp.matrix_[current_start][col] * first;
      }
    }
  }
  return det;
}

S21Matrix S21Matrix::CalcComplements() const {
  if (rows_ != cols_) throw std::logic_error("The matrix is not square");
  S21Matrix res(rows_, cols_);
  if (rows_ == 1) {
    res.matrix_[0][0] = 1;
    return res;
  }
  S21Matrix tmp(rows_ - 1, cols_ - 1);
  for (size_t i = 0; i < rows_; i++)
    for (size_t j = 0; j < rows_; j++) {
      tmp.CreateMinor(i, j, *this);
      res.matrix_[i][j] = tmp.Determinant() * pow(-1, i + j + 2);
    }
  return res;
}

S21Matrix S21Matrix::InverseMatrix() const {
  double determinant = Determinant();
  if (determinant == 0.0) throw std::logic_error("Matrix determinant is 0");
  S21Matrix tmp;
  S21Matrix res;
  tmp = CalcComplements();
  res = tmp.Transpose();
  res *= 1.0 / determinant;
  return res;
}

S21Matrix S21Matrix::ForEach(const std::function<double(const double)> &function) const {
    S21Matrix res(rows_, cols_);
    for(size_t i = 0; i < rows_; ++i)
        for(size_t j = 0; j< cols_; ++j)
            res.matrix_[i][j] = function(matrix_[i][j]);
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
  for (int i = 0; i < other.GetRows(); i++) {
    for (int j = 0; j < other.GetCols(); j++) out << other(i, j) << " ";
    out << std::endl;
  }
  return out;
}

std::istream &operator>>(std::istream &in, S21Matrix &other) {
  for (int i = 0; i < other.GetRows(); i++) {
    for (int j = 0; j < other.GetCols(); j++) in >> other(i, j);
  }
  return in;
}

void S21Matrix::CreateMinor(const size_t skip_i, const size_t skip_j,
                            const S21Matrix &other) {
  int x = 0, y = 0;
  for (int i = 0; i < other.rows_; i++) {
    for (int j = 0; j < other.rows_; j++) {
      if (i != skip_i && j != skip_j) {
        matrix_[x][y] = other.matrix_[i][j];
        y++;
      }
    }
    y = 0;
    if (i != skip_i) x++;
  }
}
}  // namespace S21
