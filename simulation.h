#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

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

  void SetMutationRate(double rate) { mutationRate_ = rate; }

private:
  const static double kCrossoverRate;
  const static double kMutationRate;
  const static double kMaxPeturbation;

  std::pair<std::vector<double>, std::vector<double>> Crossover(
    const Genome & mother,
    const Genome & father);

  std::pair<std::vector<double>, std::vector<double>> UniformCrossover(
    const Genome & mother,
    const Genome & father);

  void Mutate(std::vector<double> & genome);

  double RandomDouble() {
    std::uniform_real_distribution<double> distribution(0., 1.);
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
  double mutationRate_ = kMutationRate;
};

class SimpleSimulation {
public:
  SimpleSimulation(std::size_t msPerFrame)
    : msPerFrame_(msPerFrame) {}

  void Start() {
    lastTick_ = std::chrono::high_resolution_clock::now();

    StartImpl();
  }

  void Update(bool render) {
    if (!render) {
      UpdateImpl(render, msPerFrame_ + 1);

      lastTick_ = std::chrono::high_resolution_clock::now();
      return;
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - lastTick_;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

    if (ms.count() > msPerFrame_) {
      UpdateImpl(render, ms.count());

      lastTick_ = std::chrono::high_resolution_clock::now();
    }
  }

  void SetMsPerFrame(std::size_t ms) { msPerFrame_ = ms; }

protected:
  virtual void StartImpl() = 0;
  virtual void UpdateImpl(bool render, std::size_t ms) = 0;

private:
  std::chrono::time_point<std::chrono::steady_clock> lastTick_;
  std::size_t msPerFrame_;
};

class GenerationalSimulation {
public:
  GenerationalSimulation(std::size_t msPerFrame, std::size_t msPerGenerationRender)
    : msPerFrame_(msPerFrame), msPerGenerationRender_(msPerGenerationRender) {}

  void Start() {
    std::cout << "Generation 0\n";

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
      const uint64_t extraTicks =
        (msPerGenerationRender_ * postRenderGenerations_) / msPerFrame_;
      for (uint64_t i = 0; i < extraTicks; ++i)
        UpdateImpl(false, msPerFrame_);

      Train();

      timeSinceSpawn_ = std::chrono::milliseconds(0u);
      lastTick_ = std::chrono::high_resolution_clock::now();
    }
  }

protected:
  virtual void StartImpl() = 0;
  virtual void UpdateImpl(bool render, std::size_t ms) = 0;
  virtual void Train() = 0;

private:
  std::chrono::time_point<std::chrono::steady_clock> lastTick_;
  std::chrono::milliseconds timeSinceSpawn_;
  std::size_t msPerFrame_;
  std::size_t msPerGenerationRender_;
  std::size_t postRenderGenerations_ = 10u;
};
