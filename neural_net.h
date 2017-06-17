#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include "macros.h"

struct Neurone {
  const std::size_t size_;
  std::unique_ptr<double[]> weights_;

  Neurone(std::size_t size) : size_(size), weights_(new double [size+1]) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    for (auto i = 0u; i < size_+1; ++i)
      weights_[i] = distribution(generator);
  }
};

struct NeuroneLayer {
  const std::size_t size_;
  // TODO: could be unique_ptr?
  std::vector<Neurone> neurones_;

  NeuroneLayer(std::size_t size, std::size_t neuroneSize)
    : size_(size) {
    for (auto i = 0u; i < size; ++i)
      neurones_.emplace_back(neuroneSize);
  }
};

class NeuralNet {
public:
  const static double kThresholdBias;
  const static double kActivationResponse;

  NeuralNet(std::size_t inputs,
    std::size_t outputs,
    std::size_t hiddenLayers,
    std::size_t neuronesPerHiddenLayer)
    : inputs_(inputs)
    , outputs_(outputs)
    , hiddenLayers_(hiddenLayers)
    , neuronesPerHiddenLayer_(neuronesPerHiddenLayer) {
    Create();
  }

  std::vector<double> Process(const std::vector<double> & inputs) const {
    CHECK(inputs.size() == inputs_);

    std::vector<double> lastOutputs = inputs;
    lastOutputs.reserve(64);

    std::vector<double> outputs;
    outputs.reserve(64);

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      outputs.clear();

      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        std::size_t inputIndex = 0;
        double activation = 0.0;

        const std::size_t inputs = layers_[i].neurones_[j].size_;
        const auto & neurones = layers_[i].neurones_[j];

        for (std::size_t k = 0, length = inputs; k < length; ++k)
          activation += neurones.weights_[k] * lastOutputs[inputIndex++];

        activation += neurones.weights_[inputs] * kThresholdBias;

        outputs.push_back(ActivationFunction(activation, kActivationResponse));
      }

      lastOutputs = std::move(outputs);
      outputs.reserve(64);
    }

    return lastOutputs;
  }

  std::vector<double> GetWeights() const {
    std::vector<double> outputs;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        const auto & neurones = layers_[i].neurones_[j];
        const std::size_t inputs = neurones.size_;

        for (std::size_t k = 0, length = inputs; k < length; ++k)
          outputs.push_back(neurones.weights_[k]);

        outputs.push_back(neurones.weights_[inputs]);
      }
    }

    return outputs;
  }

  void SetWeights(const std::vector<double> & weights) const {
    auto cursor = weights.begin();

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        const auto & neurones = layers_[i].neurones_[j];
        const std::size_t inputs = neurones.size_;

        for (std::size_t k = 0, length = inputs; k < length; ++k)
          neurones.weights_[k] = *cursor++;

        neurones.weights_[inputs] = *cursor++;
      }
    }

    CHECK(cursor == weights.end());
  }

  double ActivationFunction(double activation, double response) const {
    return (1.0 / (1.0 + std::exp(-activation / response)));
  }

private:
  void Create() {
    CHECK(hiddenLayers_> 0);

    layers_.emplace_back(neuronesPerHiddenLayer_, inputs_);

    for (std::size_t i = 0, length = hiddenLayers_ - 1; i < length; ++i)
      layers_.emplace_back(neuronesPerHiddenLayer_, neuronesPerHiddenLayer_);

    layers_.emplace_back(outputs_, neuronesPerHiddenLayer_);
  }

private:
  std::size_t inputs_;
  std::size_t outputs_;
  std::size_t hiddenLayers_;
  std::size_t neuronesPerHiddenLayer_;

  std::vector<NeuroneLayer> layers_;
};

struct Genome {
  std::vector<double> weights_;
  double fitness_;

  Genome(const std::vector<double> & weights, double fitness)
    : weights_(weights), fitness_(fitness) {}

  bool operator<(const Genome & rhs) const {
    return fitness_ < rhs.fitness_;
  }
};

class Generation {
public:
  Generation(const std::vector<Genome> & population)
    : population_(population), rng_(random_()) {}

  std::vector<Genome> GetPopulation() { return population_; }

  std::vector<std::vector<double>> Evolve();

private:
  const static float kCrossoverRate;
  const static float kMutationRate;
  const static double kMaxPeturbation;

  std::pair<std::vector<double>, std::vector<double>> Crossover(
    const Genome & mother,
    const Genome & father);

  std::pair<std::vector<double>, std::vector<double>> UniformCrossover(
    const Genome & mother,
    const Genome & father);

  void Mutate(std::vector<double> & genome);

  float RandomFloat() {
    std::uniform_real_distribution<float> distribution(0., 1.);
    return distribution(rng_);
  }

  double RandomDouble(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

  int RandomInt(int min, int max) {
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(rng_);
  }

  const Genome & SelectGenome(double totalFitness);

private:
  std::vector<Genome> population_;
  std::random_device random_;
  std::mt19937 rng_;
};

class Population {
public:
  Population(std::size_t msPerFrame, std::size_t msPerGenerationRender)
    : msPerFrame_(msPerFrame), msPerGenerationRender_(msPerGenerationRender) {}

  void Start() {
    lastTick_ = std::chrono::high_resolution_clock::now();
    timeSinceSpawn_ = std::chrono::milliseconds(0u);

    StartImpl();
  }

  void Update(bool render) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - lastTick_;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

    if (!render) ms = std::chrono::milliseconds(msPerFrame_ + 1);

    if (ms.count() > msPerFrame_) {
      UpdateImpl(render, ms.count());
      timeSinceSpawn_ += ms;

      lastTick_ = std::chrono::high_resolution_clock::now();
    }

    if (timeSinceSpawn_.count() > msPerGenerationRender_) {
      // Run it for longer than shown to get to the end
      const uint64_t extraTicks = (msPerGenerationRender_ * 10) / msPerFrame_;
      for (uint64_t i = 0; i < extraTicks; ++i)
        UpdateImpl(false, msPerFrame_);

      Evolve();

      timeSinceSpawn_ = std::chrono::milliseconds(0u);
      lastTick_ = std::chrono::high_resolution_clock::now();
    }
  }

protected:
  virtual void StartImpl() = 0;
  virtual void UpdateImpl(bool render, std::size_t ms) = 0;
  virtual void Evolve() = 0;

private:
  std::chrono::time_point<std::chrono::steady_clock> lastTick_;
  std::chrono::milliseconds timeSinceSpawn_;
  std::size_t msPerFrame_;
  std::size_t msPerGenerationRender_;
};
