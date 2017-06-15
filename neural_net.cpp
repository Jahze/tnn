#include <random>
#include "neural_net.h"

const double NeuralNet::kThresholdBias = -1.0;
const double NeuralNet::kActivationResponse = 1.0;
const float Generation::kCrossoverRate = 0.7f;
const float Generation::kMutationRate = 0.15f;
const double Generation::kMaxPeturbation = 0.3;

std::vector<std::vector<double>> Generation::Evolve() {
  std::vector<std::vector<double>> nextGeneration;

  // TODO: keep elites

  const std::size_t size = population_.size();

  // TODO: better selection
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, size-1);

  while (nextGeneration.size() < size) {
    // TODO: mother and father can be the same
    const auto & father = population_[distribution(generator)];
    const auto & mother = population_[distribution(generator)];

    auto children = Crossover(father, mother);

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

  int size = mother.weights_.size();
  int crossoverPoint = RandomInt(0, size);

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
    secondborn.insert(firstborn.end(),
      father.weights_.begin(),
      father.weights_.begin() + crossoverPoint);
  }
  if (crossoverPoint < size) {
    secondborn.insert(firstborn.end(),
      mother.weights_.begin() + crossoverPoint,
      mother.weights_.end());
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
