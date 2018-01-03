#pragma once

#include <memory>
#include <random>
#include "graph.h"
#include "graphics.h"
#include "neural_net.h"
#include "simulation.h"

namespace colours {

const std::size_t BrainInputs = 3;
const std::size_t BrainOutputs = 3;
const std::size_t HiddenLayers = 1;
const std::size_t NeuronesPerLayer = 16;

struct Colour {
  double r;
  double g;
  double b;
};

class ColourObject : public ISceneObject {
public:
  ColourObject(double x, double y, const Colour & colour) {
    position_[0] = x;
    position_[1] = y;
    goal_ = colour;
  }

  void SetSize(double size) { size_ = size; }

  void Draw() const override {
    glBegin(GL_QUADS);

    glColor3d(goal_.r, goal_.g, goal_.b);
    glVertex2d(position_[0] - size_, position_[1] - size_);
    glVertex2d(position_[0] + size_, position_[1] - size_);
    glVertex2d(position_[0] + size_, position_[1] + size_);
    glVertex2d(position_[0] - size_, position_[1] + size_);

    const double halfSize = size_ * 0.5;
    glColor3d(colour_.r, colour_.g, colour_.b);
    glVertex2d(position_[0] - halfSize, position_[1] - halfSize);
    glVertex2d(position_[0] + halfSize, position_[1] - halfSize);
    glVertex2d(position_[0] + halfSize, position_[1] + halfSize);
    glVertex2d(position_[0] - halfSize, position_[1] + halfSize);

    //static const double sqrt3 = std::sqrt(3.0);
    //double r = goal_.r - colour_.r;
    //double g = goal_.g - colour_.g;
    //double b = goal_.b - colour_.b;
    //const double colour = std::sqrt(r*r + g*g + b*b) / sqrt3;

    //glColor3d(colour, colour, colour);
    //glVertex2d(position_[0] - size_, position_[1] - size_);
    //glVertex2d(position_[0] + size_, position_[1] - size_);
    //glVertex2d(position_[0] + size_, position_[1] + size_);
    //glVertex2d(position_[0] - size_, position_[1] + size_);

    glEnd();
  }

  void Update(uint64_t ms) override {
    std::vector<double> inputs{ (goal_.r * 2.0) - 1.0,
      (goal_.g * 2.0) - 1.0,
      (goal_.b * 2.0) - 1.0 };
    auto outputs = brain_.Process(inputs);
    colour_.r = outputs[0];
    colour_.g = outputs[1];
    colour_.b = outputs[2];
  }

  double CalculateFitness() const {
    double r = goal_.r - colour_.r;
    double g = goal_.g - colour_.g;
    double b = goal_.b - colour_.b;
    return 2.0 -  std::sqrt(r*r + g*g + b*b);
  }

  void SetWeights(const std::vector<double> & weights) {
    brain_.SetWeights(weights);
  }

protected:
  Colour colour_ = { 1.0, 1.0, 1.0 };
  Colour goal_ = { 1.0, 1.0, 1.0 };
  double position_[2];
  double size_ = 0.02;

  NeuralNet brain_{ BrainInputs, BrainOutputs, HiddenLayers, NeuronesPerLayer };
};

class Simulation : public ::GenerationalSimulation {
public:
  Simulation(std::size_t msPerFrame,
             std::size_t msPerGenerationRender,
             OpenGLContext & context,
             std::size_t size,
             std::size_t populationSize)
    : ::GenerationalSimulation(msPerFrame, msPerGenerationRender)
    , context_(context), rng_(random_()), goals_(size) {
    for (std::size_t i = 0; i < populationSize; ++i) {
      brains_.emplace_back(BrainInputs, BrainOutputs,
        HiddenLayers, NeuronesPerLayer);
    }
  }

  void GenerateInitialPopulation() {
    CreateSceneAndGoal();

    objects_.clear();

    const auto length = goals_.size();
    std::size_t perRow = 1;
    std::size_t square = 2;
    while (square < length) {
      square *= 2;
      perRow++;
    }

    const double cellDistance = 2.0 / (double)perRow;
    const double halfCellDistance = cellDistance / 2.0;
    std::size_t row = 0;
    std::size_t column = 0;

    for (std::size_t i = 0; i < length; ++i) {
      double x = -1.0 + halfCellDistance + (cellDistance * column++);
      double y = -1.0 + halfCellDistance + (cellDistance * row);

      if (column == perRow) {
        column = 0;
        row++;
      }

      objects_.emplace_back(new ColourObject(x, y, goals_[i]));
      auto & object = objects_.back();
      object->SetSize(halfCellDistance);
      scene_->AddObject(object.get());
    }
  }

protected:
  void StartImpl() {
    generation_ = 0;

    GenerateInitialPopulation();
  }

  void UpdateImpl(bool render, std::size_t ms) {
    scene_->Update(ms);

    if (render)
      scene_->Render(context_);
  }

  void Train() {
    std::vector<Genome> genomes;
    std::vector<double> fitnesses(objects_.size());
    const std::size_t objectCount = objects_.size();

    double bestFitness = 0.0;
    for (auto && brain : brains_) {
      auto weights = brain.GetWeights();

      for (std::size_t i = 0; i < objectCount; ++i) {
        objects_[i]->SetWeights(weights);
        objects_[i]->Update(1);
        fitnesses[i] = objects_[i]->CalculateFitness();
      }

      const double totalFitness = std::accumulate(fitnesses.begin(),
        fitnesses.end(), 0.0);
      const double avgFitness = totalFitness / (double)objectCount;
      if (bestFitness < avgFitness) bestFitness = avgFitness;
      genomes.push_back({ weights, avgFitness });
    }

    Generation generation(genomes);
    generation.SetMutationRate(0.3);

    auto nextGeneration = generation.Evolve();

    auto cursor = nextGeneration.begin();

    // FIXME: this is a hack because the first will be the best
    auto best = *cursor;

    for (auto && object : objects_)
      object->SetWeights(best);

    for (auto && brain : brains_)
      brain.SetWeights(*cursor++);

    CHECK(cursor == nextGeneration.end());

    std::cout << "Generation " << ++generation_
      << " (fitness = " << bestFitness << ")\n";
  }

private:
  void CreateGoal() {
    for (auto && goal : goals_) {
      Colour rgb = { RandomFloat(0.0, 1.0),
        RandomFloat(0.0, 1.0),
        RandomFloat(0.0, 1.0) };

      goal = rgb;
    }
  }

  void CreateSceneAndGoal() {
    scene_.reset(new Scene());

    CreateGoal();
  }

  double RandomFloat(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

private:
  OpenGLContext & context_;
  std::unique_ptr<Scene> scene_;
  std::vector<Colour> goals_;
  std::vector<NeuralNet> brains_;
  std::vector<std::unique_ptr<ColourObject>> objects_;
  std::random_device random_;
  std::mt19937 rng_;
  std::size_t generation_;
};

}
