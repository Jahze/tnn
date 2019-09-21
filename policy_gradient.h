#pragma once

#include <vector>
#include "matrix.h"
#include "neural_net.h"

class PolicyGradient {
public:
  PolicyGradient(std::size_t actionCount) : actionCount_(actionCount) {}

  void StoreIO(std::vector<double> && inputs, std::size_t action) {
    inputs_.push_back(std::move(inputs));
    actions_.push_back(action);
  }

  void StoreReward(double reward) {
    rewards_.push_back(reward);
  }

  void Reset() {
    inputs_.clear();
    actions_.clear();
    rewards_.clear();
  }

  void Teach(NeuralNet & net) {
    if (inputs_.empty()) return;

    CHECK(inputs_.size() == rewards_.size());

    // TODO: split into batches if we have a lot of inputs?
    auto outputs = net.Process(inputs_);

    const auto & rewards = DiscountedRewards();

    const std::size_t inputSize = inputs_.size();

    AlignedMatrix loss{inputSize, actionCount_};

    for (std::size_t i = 0u; i < inputSize; ++i) {
      for (std::size_t j = 0u; j < actionCount_; ++j) {
        double ideal = j == actions_[i] ? 1.0 : 0.0;
        loss[i][j] = (outputs[i][j] - ideal) * rewards[i];
      }
    }

    net.BackPropagationCrossEntropy(loss);
  }

  void UpdateLastReward(double reward) {
    rewards_.back() = reward;
  }

  size_t FrameCount() {
    return inputs_.size();
  }

  std::vector<double> Inputs(size_t frame) {
    CHECK(frame < FrameCount());
    return inputs_[frame];
  }

  size_t SelectedAction(size_t frame) {
    CHECK(frame < FrameCount());
    return actions_[frame];
  }

  double Reward(size_t frame) {
    CHECK(frame < FrameCount());
    auto rewards = DiscountedRewards();
    return rewards[frame];
  }

private:
  std::vector<double> DiscountedRewards() const {
    const std::size_t inputSize = inputs_.size();

    std::vector<double> out(inputSize);

    const double Gamma = 0.95;
    double reward = 0.0;
    double mean = 0.0;

    for (std::size_t i = inputSize; i > 0u; --i) {
      reward = reward * Gamma + rewards_[i - 1u];
      out[i - 1u] = reward;
      mean += reward;
    }

    mean /= inputSize;

    double stddev = 0.0;

    for (std::size_t i = 0u; i < inputSize; ++i)
      stddev += std::pow(out[i] - mean, 2.);

    stddev = std::sqrt(stddev / inputSize);

    // This can happen if the whole run generates no rewards
    if (stddev == 0.0) {
      return out;
    }

    for (std::size_t i = 0u; i < inputSize; ++i) {
      out[i] -= mean;
      out[i] /= stddev;
    }

    return out;
  }

private:
  std::size_t actionCount_;
  std::vector<std::vector<double>> inputs_;
  std::vector<std::size_t> actions_;
  std::vector<double> rewards_;
};
