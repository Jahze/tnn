#include <algorithm>
#include <numeric>
#include <random>
#include "neural_net.h"

const double NeuralNet::kThresholdBias = -1.0;
const double NeuralNet::kActivationResponse = 1.0;
const float Generation::kCrossoverRate = 0.7f;
const float Generation::kMutationRate = 0.15f;
const double Generation::kMaxPeturbation = 0.3;

std::vector<std::vector<double>> Generation::Evolve() {
  std::vector<std::vector<double>> nextGeneration;

  std::sort(population_.begin(), population_.end());

  // TODO: keep elites

  const std::size_t size = population_.size();

  auto AccumulateFitness =
    [](double accum, const Genome & genome)
    { return accum + genome.fitness_; };

  double totalFitness = std::accumulate(
    population_.begin(), population_.end(),
    0.0, AccumulateFitness);

  // TODO: better selection
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<std::size_t> distribution(0, size-1);

  while (nextGeneration.size() < size) {
    // TODO: mother and father can be the same
    const auto & father = SelectGenome(totalFitness);
    const auto & mother = SelectGenome(totalFitness);

    //auto children = Crossover(father, mother);
    auto children = UniformCrossover(father, mother);

    Mutate(children.first);
    Mutate(children.second);

    nextGeneration.push_back(children.first);
    nextGeneration.push_back(children.second);
  }

  return nextGeneration;
}

std::pair<std::vector<double>, std::vector<double>> Generation::Crossover(
  const Genome & mother,
  const Genome & father) {
  if (RandomFloat() > kCrossoverRate) {
    return std::make_pair(mother.weights_, father.weights_);
  }

  int size = static_cast<int>(mother.weights_.size());
  int crossoverPoint = RandomInt(0, size);

  // TODO is this correct if cp == size?
  std::vector<double> firstborn;
  if (crossoverPoint > 0) {
    firstborn.insert(firstborn.end(),
      mother.weights_.begin(),
      mother.weights_.begin() + crossoverPoint);
  }
  if (crossoverPoint < size) {
    firstborn.insert(firstborn.end(),
      father.weights_.begin() + crossoverPoint,
      father.weights_.end());
  }

  std::vector<double> secondborn;
  if (crossoverPoint > 0) {
    secondborn.insert(secondborn.end(),
      father.weights_.begin(),
      father.weights_.begin() + crossoverPoint);
  }
  if (crossoverPoint < size) {
    secondborn.insert(secondborn.end(),
      mother.weights_.begin() + crossoverPoint,
      mother.weights_.end());
  }

  return { firstborn, secondborn };
}

std::pair<std::vector<double>, std::vector<double>>
  Generation::UniformCrossover(
  const Genome & mother,
  const Genome & father) {
  if (RandomFloat() > kCrossoverRate) {
    return std::make_pair(mother.weights_, father.weights_);
  }

  std::vector<double> firstborn, secondborn;
  for (std::size_t i = 0u, length = mother.weights_.size(); i < length; ++i) {
    if (RandomFloat() < 0.5f) {
      firstborn.push_back(mother.weights_[i]);
      secondborn.push_back(father.weights_[i]);
    }
    else {
      firstborn.push_back(father.weights_[i]);
      secondborn.push_back(mother.weights_[i]);
    }
  }

  return { firstborn, secondborn };
}

void Generation::Mutate(std::vector<double> & genome) {
  for (auto & gene : genome) {
    if (RandomFloat() > kMutationRate)
      continue;

    gene += RandomDouble(-1.0, 1.0) * kMaxPeturbation;
  }
}

const Genome & Generation::SelectGenome(double totalFitness) {
  double target = RandomDouble(0.0, totalFitness);
  double accumulatedFitness = 0.0;

  for (auto && genome : population_) {
    accumulatedFitness += genome.fitness_;
    if (accumulatedFitness > target)
      return genome;
  }

  CHECK(false);
  return population_.back();
}
