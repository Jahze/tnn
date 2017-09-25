#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

template<typename T>
class MovingAverage {
public:
  MovingAverage(std::size_t samples) : samples_(samples) {}

  void AddDataPoint(T v) {
    if (index_ == std::size_t(-1)) {
      data_.resize(samples_, v);
      index_ = 1;
      return;
    }

    if (index_ == samples_)
      index_ = 0u;

    data_[index_++] = v;
  }

  T Average() const {
    return std::accumulate(data_.begin(), data_.end(), T())
      / static_cast<T>(samples_);
  }

private:
  std::vector<T> data_;
  std::size_t samples_;
  std::size_t index_ = std::size_t(-1);
};
