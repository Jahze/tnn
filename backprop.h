#pragma once

#include <memory>
#include <random>
#include "graph.h"
#include "graphics.h"
#include "neural_net.h"

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

protected:
  Colour colour_ = { 1.0, 1.0, 1.0 };
  Colour goal_ = { 1.0, 1.0, 1.0 };
  double position_[2];
  double size_ = 0.02;

  NeuralNet brain_{ BrainInputs, BrainOutputs, HiddenLayers, NeuronesPerLayer };
};

class Simulation : public ::Simulation {
public:
  Simulation(std::size_t msPerFrame,
             std::size_t msPerGenerationRender,
             OpenGLContext & context,
             std::size_t trainingDataSize)
    : ::Simulation(msPerFrame, msPerGenerationRender)
    , context_(context), rng_(random_()), trainingData_(trainingDataSize)
    , brain_(BrainInputs, BrainOutputs, HiddenLayers, NeuronesPerLayer) {}

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
    const std::size_t objectCount = objects_.size();

    auto weights = brain_.GetWeights();

    for (std::size_t i = 0; i < objectCount; ++i) {
      const auto & inputs = objects_[i]->GetInputs();
      const auto & outputs = objects_[i]->GetIdealOutputs();

      auto lossFunction = [&outputs](double value, std::size_t i) {
        return -(outputs[i] - value);
      };

      brain_.BackPropagation(inputs, lossFunction);
    }

    for (auto && object : objects_)
      object->SetWeights(brain_.GetWeights());

    std::cout << "Generation " << ++generation_ << "\n";
  }

private:
  double RandomDouble(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

private:
  OpenGLContext & context_;
  std::unique_ptr<Scene> scene_;
  std::vector<Colour> trainingData_;
  NeuralNet brain_;
  std::vector<std::unique_ptr<ColourObject>> objects_;
  std::random_device random_;
  std::mt19937 rng_;
  std::size_t generation_;
};

}
