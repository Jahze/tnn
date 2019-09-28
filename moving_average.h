#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

template<typename T>
class MovingAverage {
public:
  static_assert(
      std::is_arithmetic_v<T>,
      "MovingAverage<T> requires arithmetic type");

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

  T Max() const {
    T max;

    if constexpr (std::is_floating_point_v<T>)
      max = -std::numeric_limits<T>::max();
    else
      max = std::numeric_limits<T>::min();

    for (auto datum : data_) {
      if (datum > max)
        max = datum;
    }

    return max;
  }

private:
  std::vector<T> data_;
  std::size_t samples_;
  std::size_t index_ = std::size_t(-1);
};
