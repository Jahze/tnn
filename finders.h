#pragma once

#include <memory>
#include <random>
#include "graphics.h"
#include "neural_net.h"

class SimpleObject : public ISceneObject {
public:
  SimpleObject(double x, double y) {
    position_[0] = x;
    position_[1] = y;
  }

  void SetSize(double size) { size_ = size; }

  void SetColour(double r, double g, double b) {
    colour_[0] = r; colour_[1] = g; colour_[2] = b;
  }

  void Draw() const override {
    glBegin(GL_QUADS);

    glColor3dv(colour_);
    glVertex2d(position_[0] - size_, position_[1] - size_);
    glVertex2d(position_[0] + size_, position_[1] - size_);
    glVertex2d(position_[0] + size_, position_[1] + size_);
    glVertex2d(position_[0] - size_, position_[1] + size_);

    glEnd();
  }

  void Update(uint64_t ms) override {
  }

protected:
  double colour_[3] = { 1.0, 1.0, 1.0 };
  double position_[2];
  double size_ = 0.02;
};

class SmartObject : public SimpleObject {
public:
  SmartObject(double x, double y, double goalx, double goaly)
    : SimpleObject(x, y) {
    goal_[0] = goalx;
    goal_[1] = goaly;
  }

  void Update(uint64_t ms) override {
    static const double kSpeed = 0.02;

    std::vector<double> inputs{ position_[0], position_[1], goal_[0], goal_[1] };
    auto outputs = brain_.Process(inputs);

    //const float xSpeed = (static_cast<float>(outputs[0]) +
    //  (-1.f * static_cast<float>(outputs[1]))) * kSpeed *
    //  static_cast<float>(outputs[2]);
    //const float ySpeed = (static_cast<float>(outputs[3]) +
    //  (-1.f * static_cast<float>(outputs[4]))) * kSpeed *
    //  static_cast<float>(outputs[5]);

    double xSpeed = outputs[0] + (-1.f * outputs[1]);
    double ySpeed = outputs[2] + (-1.f * outputs[3]);

    //double xSpeed = outputs[0] > outputs[1] ? kSpeed : -kSpeed;
    //double ySpeed = outputs[2] > outputs[3] ? kSpeed : -kSpeed;

    xSpeed *= outputs[4] * kSpeed * ms;
    ySpeed *= outputs[5] * kSpeed * ms;

    //if (position_[0] + xSpeed - size_ < -1.0)
    //  return;

    //if (position_[1] + ySpeed - size_ < -1.0)
    //  return;

    //if (position_[0] + xSpeed + size_ > 1.0)
    //  return;

    //if (position_[1] + ySpeed + size_ > 1.0)
    //  return;

    position_[0] += xSpeed;
    position_[1] += ySpeed;
  }

  double CalculateFitness() const {
    const double x = position_[0] - goal_[0];
    const double y = position_[1] - goal_[1];
    return accumulatedFitness_ + 1.0 / std::sqrt(x*x + y*y);
  }

  Genome GetGenome() const {
    return { brain_.GetWeights(), CalculateFitness() };
  }

  void SetWeights(const std::vector<double> & weights) {
    brain_.SetWeights(weights);
  }

  void SetGoal(double x, double y) {
    goal_[0] = x; goal_[1] = y;
    accumulatedFitness_ = CalculateFitness();
  }

private:
  double goal_[2];
  double accumulatedFitness_ = 0.0;

  const static std::size_t brainInputs = 4;
  const static std::size_t brainOutputs = 6;

  NeuralNet brain_{ brainInputs, brainOutputs, 1, 16 };
};

class Population {
public:
  Population() : rng_(random_()) {}

  Scene * GenerateInitialPopulation() {
    auto goal = CreateSceneAndGoal();

    objects_.clear();
    for (std::size_t i = 0; i < kPopulationSize; ++i) {
      objects_.emplace_back(new SmartObject(0.0, 0.0, goal.first, goal.second));
      scene_->AddObject(objects_.back().get());
    }

    return scene_.get();
  }

  Scene * Evolve() {
    std::vector<Genome> genomes;
    for (auto && object : objects_)
      genomes.push_back(object->GetGenome());

    Generation generation(genomes);

    auto nextGeneration = generation.Evolve();

    auto goal = CreateSceneAndGoal();

    objects_.clear();
    auto cursor = nextGeneration.begin();

    for (std::size_t i = 0; i < kPopulationSize; ++i) {
      objects_.emplace_back(new SmartObject(0.0, 0.0, goal.first, goal.second));
      objects_.back()->SetWeights(*cursor++);
      scene_->AddObject(objects_.back().get());
    }

    CHECK(cursor == nextGeneration.end());

    return scene_.get();
  }

  void CreateNewGoal() {
    scene_->RemoveObject(goal_.get());

    auto goal = CreateGoal();

    for (auto && object : objects_)
      object->SetGoal(goal.first, goal.second);
  }

  std::vector<std::unique_ptr<SmartObject>>::iterator begin()
    { return objects_.begin(); }

  std::vector<std::unique_ptr<SmartObject>>::iterator end()
    { return objects_.end(); }


private:
  const std::size_t kPopulationSize = 50u;

  std::pair<double, double> CreateGoal() {
    const double goalX = RandomFloat(-0.9, 0.9);
    const double goalY = RandomFloat(-0.9, 0.9);

    goal_.reset(new SimpleObject(goalX, goalY));
    goal_->SetSize(0.01);
    goal_->SetColour(0.0, 1.0, 1.0);
    scene_->AddObject(goal_.get());

    return std::make_pair(goalX, goalY);
  }

  std::pair<double, double> CreateSceneAndGoal() {
    scene_.reset(new Scene());

    return CreateGoal();
  }

  double RandomFloat(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

private:
  std::unique_ptr<Scene> scene_;
  std::unique_ptr<SimpleObject> goal_;
  std::vector<std::unique_ptr<SmartObject>> objects_;
  std::random_device random_;
  std::mt19937 rng_;
};
