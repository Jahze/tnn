#pragma once

#include <memory>
#include <random>
#include "graphics.h"
#include "neural_net.h"
#include "simulation.h"

namespace finders {

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
    static const double kSpeed = 0.01;

    std::vector<double> inputs{ position_[0], position_[1], goal_[0], goal_[1] };
    auto outputs = brain_.Process(inputs);

    double xSpeed = (outputs[0] * 2.0) - 1.0;
    double ySpeed = (outputs[1] * 2.0) - 1.0;

    xSpeed *= outputs[2] * kSpeed * ms;
    ySpeed *= outputs[3] * kSpeed * ms;

    //double angle = TWO_PI * outputs[0];
    //double xSpeed = std::cos(angle);
    //double ySpeed = std::sin(angle);

    //xSpeed *= outputs[1] * kSpeed * ms;
    //ySpeed *= outputs[2] * kSpeed * ms;

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
  const static std::size_t brainOutputs = 4;

  NeuralNet brain_{ brainInputs, brainOutputs, 1, 8 };
};

class Simulation : public ::GenerationalSimulation {
public:
  Simulation(std::size_t msPerFrame,
             std::size_t msPerGenerationRender,
             OpenGLContext & context,
             std::size_t goals)
    : ::GenerationalSimulation(msPerFrame, msPerGenerationRender)
    , context_(context), rng_(random_()), goals_(goals) {}

  void GenerateInitialPopulation() {
    auto goal = CreateSceneAndGoal();

    objects_.clear();
    for (std::size_t i = 0; i < kPopulationSize; ++i) {
      objects_.emplace_back(new SmartObject(0.0, 0.0, goal.first, goal.second));
      scene_->AddObject(objects_.back().get());
    }
  }

  void CreateNewGoal() {
    scene_->RemoveObject(goal_.get());

    auto goal = CreateGoal();

    for (auto && object : objects_)
      object->SetGoal(goal.first, goal.second);
  }

protected:
  void StartImpl() {
    generation_ = 0;
    currentGoal_ = 0;

    GenerateInitialPopulation();
  }

  void UpdateImpl(bool render, std::size_t ms) {
    scene_->Update(ms);

    if (render) {
      double fitness = 0.0;
      SmartObject * best = nullptr;
      for (auto && object : objects_) {
        double f = object->CalculateFitness();
        if (f > fitness) {
          best = object.get();
          fitness = f;
        }
      }

      best->SetColour(1.0, 0.0, 0.0);

      scene_->Render(context_);

      best->SetColour(1.0, 1.0, 1.0);
    }
  }

  void Train() {
    if (++currentGoal_ < goals_) {
      CreateNewGoal();
    }
    else {
      DoEvolve();

      std::cout << "Generation " << ++generation_ << "\n";

      currentGoal_ = 0u;
    }
  }

  void DoEvolve() {
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
  }

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
  OpenGLContext & context_;
  std::unique_ptr<Scene> scene_;
  std::unique_ptr<SimpleObject> goal_;
  std::vector<std::unique_ptr<SmartObject>> objects_;
  std::random_device random_;
  std::mt19937 rng_;
  const std::size_t goals_;
  std::size_t currentGoal_;
  std::size_t generation_;
};

}