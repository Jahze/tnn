#pragma once

#include <memory>
#include <random>
#include "graph.h"
#include "graphics.h"
#include "moving_average.h"
#include "neural_net.h"
#include "simulation.h"
#include "timer.h"

namespace backprop {

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

    glEnd();
  }

  void Update(uint64_t ms) override {
    std::vector<double> inputs = GetInputs();
    auto outputs = brain_.Process(inputs);
    colour_.r = outputs[0];
    colour_.g = outputs[1];
    colour_.b = outputs[2];
  }

  void SetWeights(const std::vector<double> & weights) {
    brain_.SetWeights(weights);
  }

  std::vector<double> GetInputs() const {
    return { (goal_.r * 2.0) - 1.0,
      (goal_.g * 2.0) - 1.0,
      (goal_.b * 2.0) - 1.0 };
  }

  std::vector<double> GetIdealOutputs() const {
    return { goal_.r, goal_.g, goal_.b };
  }

  const Colour & GetColour() const { return colour_; }

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
             std::size_t trainingDataSize)
    : ::GenerationalSimulation(msPerFrame, msPerGenerationRender)
    , context_(context), rng_(random_()), trainingData_(trainingDataSize)
    , brain_(BrainInputs, BrainOutputs, HiddenLayers, NeuronesPerLayer)
    , average_(10) {

    // INVESTIGATE: ReLu activation (even with lr of 0.001) stops this
    // network from converging

    //brain_.SetHiddenLayerActivationType(ActivationType::ReLu);

    brain_.SetOptimiser(Optimiser::AdamOptimiser);
    brain_.SetLearningRate(0.01);
  }

protected:
  void StartImpl() {
    generation_ = 0;

    scene_.reset(new Scene());

    for (auto && datum : trainingData_) {
      Colour rgb = { RandomDouble(0.0, 1.0),
        RandomDouble(0.0, 1.0),
        RandomDouble(0.0, 1.0) };

      datum = rgb;
    }

    const auto length = trainingData_.size();
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

      objects_.emplace_back(new ColourObject(x, y, trainingData_[i]));
      auto & object = objects_.back();
      object->SetSize(halfCellDistance);
      scene_->AddObject(object.get());
    }
  }

  void UpdateImpl(bool render, std::size_t ms) {
    scene_->Update(ms);

    if (render)
      scene_->Render(context_);
  }

  void Train() {
    Timer timer;

    double totalLoss = 0.0;

    std::vector<std::vector<double>> batchInputs;
    std::vector<std::vector<double>> batchIdealOutputs;

    for (auto && object : objects_) {
      const auto & inputs = object->GetInputs();
      const auto & outputs = object->GetIdealOutputs();
      const Colour & colour = object->GetColour();

      double rloss = colour.r - outputs[0];
      double gloss = colour.g - outputs[1];
      double bloss = colour.b - outputs[2];

      totalLoss += std::sqrt(rloss*rloss + gloss*gloss + bloss*bloss);

      batchInputs.push_back(inputs);
      batchIdealOutputs.push_back(outputs);

      if (batchInputs.size() == BatchSize) {
        brain_.BackPropagationThreaded(batchInputs, batchIdealOutputs);
        batchInputs.clear();
        batchIdealOutputs.clear();
      }
    }

    if (!batchInputs.empty())
      brain_.BackPropagationThreaded(batchInputs, batchIdealOutputs);

    for (auto && object : objects_)
      object->SetWeights(brain_.GetWeights());

    average_.AddDataPoint(timer.ElapsedMicroseconds());

    std::cout << "Rolling average generation time = "
      << average_.Average() << "us\n";

    std::cout << "Generation " << ++generation_
      << " (loss=" << totalLoss << ")\n";
  }

private:
  double RandomDouble(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

private:
  constexpr static size_t BatchSize = 8;

  OpenGLContext & context_;
  std::unique_ptr<Scene> scene_;
  std::vector<Colour> trainingData_;
  NeuralNet brain_;
  std::vector<std::unique_ptr<ColourObject>> objects_;
  std::random_device random_;
  std::mt19937 rng_;
  std::size_t generation_;
  MovingAverage<uint64_t> average_;
};

}
