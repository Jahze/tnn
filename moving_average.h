#pragma once

#include <cstddef>
#include <numeric>
#include <unordered_map>
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

  void PrintHistogram() {
    if constexpr (std::is_floating_point_v<T>)
      PrintHistogramImpl<int64_t>();
    else
      PrintHistogramImpl<T>();
  }

private:
  template<typename BucketType>
  void PrintHistogramImpl() {
    static_assert(std::is_integral_v<BucketType>, "BucketType not integral");

    std::unordered_map<BucketType,size_t> buckets;

    for (auto datum : data_) {
      // If buckets[bucket] doesn't exist operator[] will value-initialise it.

      if constexpr (std::is_floating_point_v<T>) {
        BucketType bucket = static_cast<BucketType>(std::floor(datum + 0.5));
        buckets[bucket] += 1;
      }
      else {
        buckets[datum] += 1;
      }
    }

    const auto minmaxBucket = std::minmax_element(
        std::begin(buckets),
        std::end(buckets),
        [](const auto & lhs, const auto & rhs) {
          return lhs.first < rhs.first;
        });

    const BucketType minBucket = minmaxBucket.first->first;
    const BucketType maxBucket = minmaxBucket.second->first;

    const size_t bucketTextLength = std::max(
        std::to_string(minBucket).length(),
        std::to_string(maxBucket).length());

    const size_t countTextLength = std::to_string(samples_).length();

    constexpr static size_t HistogramBarMaximumLength = 50ull;

    for (BucketType i = minBucket; i <= maxBucket; ++i) {
      const std::string bucket = std::to_string(i);
      const std::string count = std::to_string(buckets[i]);

      const double percent =
          static_cast<double>(buckets[i]) /
          static_cast<double>(samples_);

      const size_t pips = std::floor(percent * HistogramBarMaximumLength + 0.5);

      std::cout
          << ' '
          << std::string(bucketTextLength - bucket.length(), ' ')
          << bucket
          << " ("
          << std::string(countTextLength - count.length(), ' ')
          << count
          << ") | "
          << (buckets[i] > 0ull ? '\xDB' : ' ')
          << std::string(pips, '\xDB')
          << "\n";
    }
  }

private:
  std::vector<T> data_;
  std::size_t samples_;
  std::size_t index_ = std::size_t(-1);
};
