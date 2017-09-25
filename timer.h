#pragma once

#include <chrono>
#include <cstdint>

class Timer {
public:
  Timer() {
    Reset();
  }

  double ElapsedSeconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
      (now - start_);
    uint64_t ms = static_cast<uint64_t>(elapsed.count());
    return static_cast<double>(ms) / 1000.0;
  }

  uint64_t ElapsedMicroseconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>
      (now - start_);
    return static_cast<uint64_t>(elapsed.count());
  }

  void Reset() {
    start_ = std::chrono::high_resolution_clock::now();
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
};
