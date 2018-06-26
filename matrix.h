#pragma once

#include <immintrin.h>
#include "threadpool.h"

template<typename T>
inline std::size_t AlignTo32Bytes(std::size_t size) {
  return size + (32 - ((size * sizeof(T)) % 32)) / sizeof(T);
}

// This storage is aligned on a 32 byte boundary as well as being a multiple of
// 32 bytes. This means it can be used in AVX instructions and always fill an
// entire register, making batching easier by not having to deal with a last
// batch that doesn't fill a register.

template<typename T>
class Aligned32ByteRAIIStorage {
public:
  Aligned32ByteRAIIStorage() : storage_(nullptr), size_(0), raw_(nullptr) {}

  Aligned32ByteRAIIStorage(std::size_t size) {
    size_ = AlignTo32Bytes<T>(size);
    AlignedNew();
  }

  ~Aligned32ByteRAIIStorage() { delete [] raw_; }

  Aligned32ByteRAIIStorage(const Aligned32ByteRAIIStorage<T> &) = delete;
  Aligned32ByteRAIIStorage& operator=(
    const Aligned32ByteRAIIStorage<T> &) = delete;

  Aligned32ByteRAIIStorage(Aligned32ByteRAIIStorage && rhs) {
    storage_ = rhs.storage_;
    raw_ = rhs.raw_;
    size_ = rhs.size_;
    rhs.storage_ = nullptr;
    rhs.raw_ = nullptr;
  }

  Aligned32ByteRAIIStorage& operator=(Aligned32ByteRAIIStorage && rhs) {
    storage_ = rhs.storage_;
    raw_ = rhs.raw_;
    size_ = rhs.size_;
    rhs.storage_ = nullptr;
    rhs.raw_ = nullptr;
  }

  T & operator[](std::size_t i) { return storage_[i]; }
  T operator[](std::size_t i) const { return storage_[i]; }
  T * Get() { return storage_; }
  const T * Get() const { return storage_; }

  void Reset(std::size_t size) {
    // Don't resize unless we need more space
    if (size < size_) return;

    delete [] raw_;

    size_ = AlignTo32Bytes<T>(size);
    AlignedNew();
  }

  // TODO: debugging temp
  std::size_t Size() const { return size_; }

private:
  void AlignedNew() {
    const std::size_t elements = size_ + (32 / sizeof(T));
    raw_ = new T[elements];
    storage_ = raw_;
    while (reinterpret_cast<uint64_t>(storage_) % 32 != 0)
      storage_++;
  }

private:
  T * raw_;
  T * storage_;
  std::size_t size_;
};

inline void SIMDMultiply(
  const double * lhs,
  const double * rhs,
  Aligned32ByteRAIIStorage<double> & dest,
  std::size_t size) {

  const std::size_t batches = AlignTo32Bytes<double>(size) / 4;
  for (std::size_t k = 0; k < batches; ++k) {
    __m256d l = _mm256_load_pd(lhs + (k * 4));
    __m256d r = _mm256_load_pd(rhs + (k * 4));
    __m256d result = _mm256_mul_pd(l, r);
    std::memcpy(dest.Get() + (k * 4),
      result.m256d_f64, sizeof(double) * 4);
  }
}

struct AlignedMatrix {
  std::size_t rows_;
  std::size_t columns_;
  std::size_t alignedColumns_;
  Aligned32ByteRAIIStorage<double> values_;

  AlignedMatrix() : rows_(0u), columns_(0u) {}

  AlignedMatrix(std::size_t rows, std::size_t columns) {
    Reset(rows, columns);
  }

  void Reset(std::size_t rows, std::size_t columns) {
    rows_ = rows;
    columns_ = columns;
    alignedColumns_ = AlignTo32Bytes<double>(columns_);
    values_.Reset(rows_ * alignedColumns_);

    // Initialise the unused values allocated for alignment to 0 so
    // that they don't contribute to the multiplication
    Zero();
  }

  void Zero() {
    std::memset(values_.Get(), 0, rows_ * alignedColumns_ * sizeof(double));
  }

  void Transpose(AlignedMatrix & out) {
    out.Reset(columns_, rows_);

    for (std::size_t i = 0u; i < rows_; ++i) {
      double * row = Row(i);
      for (std::size_t j = 0u; j < columns_; ++j) {
        out.Value(j, i) = row[j];
      }
    }
  }

  double * Row(std::size_t row) {
    return values_.Get() + (row * alignedColumns_);
  }

  const double * Row(std::size_t row) const {
    return values_.Get() + (row * alignedColumns_);
  }

  double & Value(std::size_t row, std::size_t column) {
    return Row(row)[column];
  }

  double Value(std::size_t row, std::size_t column) const {
    return Row(row)[column];
  }

  void Multiply(const Aligned32ByteRAIIStorage<double> & inputs,
      Aligned32ByteRAIIStorage<double> & outputs) const {
    std::size_t weightsIndex = 0;

    for (std::size_t i = 0u; i < rows_; ++i) {
      __m256d result = _mm256_setzero_pd();

      const double * row = Row(i);
      const double * inputPtr = inputs.Get();

      const std::size_t batches = alignedColumns_ / 4;
      for (std::size_t j = 0u; j < batches; ++j) {
        std::size_t stride = j * 4;
        result = _mm256_add_pd(result, _mm256_mul_pd(
          _mm256_load_pd(inputPtr + stride),
          _mm256_load_pd(row + stride)));
      }

      result = _mm256_hadd_pd(result, result);

      outputs[i] = result.m256d_f64[0] + result.m256d_f64[2];
    }
  }

  void MultiplyThreaded(double * inputs, double * outputs) const {
    ThreadPool * pool = GetCpuSizedThreadPool();

    BatchTasks tasks(*pool);
    tasks.CreateBatches(rows_,
      [this, inputs, outputs](std::size_t start, std::size_t end) {

      const double * values = Row(start);

      for (std::size_t i = start; i < end; ++i) {
        __m256d result = _mm256_setzero_pd();

        const std::size_t batches = alignedColumns_ / 4;
        for (std::size_t j = 0u; j < batches; ++j) {
          std::size_t stride = j * 4;
          result = _mm256_add_pd(result, _mm256_mul_pd(
            _mm256_load_pd(inputs + stride),
            _mm256_load_pd(values + stride)));
        }

        result = _mm256_hadd_pd(result, result);

        outputs[i] = result.m256d_f64[0] + result.m256d_f64[2];

        values += alignedColumns_;
      }
    });

    tasks.Run();
  }

  void Subtract(AlignedMatrix & rhs) {
    for (std::size_t i = 0u; i < rows_; ++i) {

      double * row = Row(i);
      const double * input = rhs.Row(i);

      for (std::size_t j = 0u; j < columns_; ++j)
        row[j] -= input[j];
    }
  }

  void SubtractThreaded(AlignedMatrix & rhs) {
    ThreadPool * pool = GetCpuSizedThreadPool();

    BatchTasks tasks(*pool);
    tasks.CreateBatches(rows_,
      [this, &rhs](std::size_t start, std::size_t end) {

      double * values = Row(start);
      const double * inputs = rhs.Row(start);

      for (std::size_t i = start; i < end; ++i) {
        __m256d result = _mm256_setzero_pd();

        const std::size_t batches = alignedColumns_ / 4;
        for (std::size_t j = 0u; j < batches; ++j) {
          std::size_t stride = j * 4;
          result = _mm256_sub_pd(
            _mm256_load_pd(values + stride),
            _mm256_load_pd(inputs + stride));

          std::memcpy(values + stride, result.m256d_f64, sizeof(double) * 4);
        }

        values += alignedColumns_;
        inputs += alignedColumns_;
      }
    });

    tasks.Run();
  }
};
