#pragma once

#include <memory>
#include <random>
#include "graphics.h"
#include "neural_net.h"

namespace chasers {

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

  double GetX() const { return position_[0]; }
  double GetY() const { return position_[1]; }

protected:
  double colour_[3] = { 1.0, 1.0, 1.0 };
  double position_[2];
  double size_ = 0.02;
};

class GoalObject : public SimpleObject {
public:
  GoalObject(std::mt19937 & rng, double x, double y)
    : SimpleObject(x, y), rng_(rng) {
    ChangeSpeed();
  }

  void Update(uint64_t ms) override {
    static const double kSpeed = 0.005;
    static const double kChangeChance = 0.02;

    if (changeDistribution_(rng_) < kChangeChance)
      ChangeSpeed();

    position_[0] += speed_[0] * kSpeed;
    position_[1] += speed_[1] * kSpeed;

    if (position_[0] > 1.0) {
      position_[0] = 1.0;
      if (speed_[0] > 0.0) speed_[0] = -speed_[0];
    }

    if (position_[0] < -1.0) {
      position_[0] = -1.0;
      if (speed_[0] < 0.0) speed_[0] = -speed_[0];
    }

    if (position_[1] > 1.0) {
      position_[1] = 1.0;
      if (speed_[1] > 0.0) speed_[1] = -speed_[1];
    }

    if (position_[1] < -1.0) {
      position_[1] = -1.0;
      if (speed_[1] < 0.0) speed_[1] = -speed_[1];
    }
  }

private:
  void ChangeSpeed() {
    speed_[0] = speedDistribution_(rng_);
    speed_[1] = speedDistribution_(rng_);
  }

private:
  std::mt19937 & rng_;
  std::uniform_real_distribution<> speedDistribution_{-1.0, 1.0};
  std::uniform_real_distribution<> changeDistribution_{0.0, 1.0};
  double speed_[2];
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

    double xSpeed = outputs[0] + (-1.f * outputs[1]);
    double ySpeed = outputs[2] + (-1.f * outputs[3]);

    xSpeed *= outputs[4] * kSpeed * ms;
    ySpeed *= outputs[5] * kSpeed * ms;

    position_[0] += xSpeed;
    position_[1] += ySpeed;
  }

  double CalculateDistance() const {
    const double x = position_[0] - goal_[0];
    const double y = position_[1] - goal_[1];
    return std::sqrt(x*x + y*y);
  }

  //void AccumulateDistance() { accumulatedDistance_ += CalculateDistance(); }
  //double GetAccumulatedDistance() const { return accumulatedDistance_; }

  Genome GetGenome(double maxDistance) const {
    //return { brain_.GetWeights(), maxDistance + 1.0 - accumulatedDistance_ };
    return{ brain_.GetWeights(), maxDistance + 1.0 - CalculateDistance() };
  }

  void SetWeights(const std::vector<double> & weights) {
    brain_.SetWeights(weights);
  }

  void SetGoal(double x, double y) { goal_[0] = x; goal_[1] = y; }

private:
  double goal_[2];
  double accumulatedDistance_ = 0.0;

  const static std::size_t brainInputs = 4;
  const static std::size_t brainOutputs = 6;

  NeuralNet brain_{ brainInputs, brainOutputs, 1, 16 };
};

class Population : public ::Population {
public:
  Population(std::size_t msPerFrame,
             std::size_t msPerGenerationRender,
             OpenGLContext & context)
    : ::Population(msPerFrame, msPerGenerationRender)
    , context_(context), rng_(random_()) {}

  void GenerateInitialPopulation() {
    auto goal = CreateSceneAndGoal();

    objects_.clear();
    for (std::size_t i = 0; i < kPopulationSize; ++i) {
      objects_.emplace_back(new SmartObject(0.0, 0.0, goal.first, goal.second));
      scene_->AddObject(objects_.back().get());
    }
  }

protected:
  void StartImpl() {
    generation_ = 0;

    GenerateInitialPopulation();
  }

  void UpdateImpl(bool render, std::size_t ms) {
    for (auto && object : objects_) {
      object->SetGoal(goal_->GetX(), goal_->GetY());
      //object->AccumulateDistance();
    }

    scene_->Update(ms);

    if (render) {
      double minDistance = std::numeric_limits<double>::max();
      SmartObject * best = nullptr;
      for (auto && object : objects_) {
        double d = object->CalculateDistance();
        if (d < minDistance) {
          best = object.get();
          minDistance = d;
        }
      }

      best->SetColour(1.0, 0.0, 0.0);

      scene_->Render(context_);

      best->SetColour(1.0, 1.0, 1.0);
    }
  }

  void Evolve() {
    DoEvolve();

    std::cout << "Generation " << ++generation_ << "\n";
  }

  void DoEvolve() {
    double maxDistance = 0.0;
    for (auto && object : objects_) {
      //double d = object->GetAccumulatedDistance();
      double d = object->CalculateDistance();
      if (d > maxDistance) maxDistance = d;
    }

    std::vector<Genome> genomes;
    for (auto && object : objects_)
      genomes.push_back(object->GetGenome(maxDistance));

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

    goal_.reset(new GoalObject(rng_, goalX, goalY));
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
  std::unique_ptr<GoalObject> goal_;
  std::vector<std::unique_ptr<SmartObject>> objects_;
  std::random_device random_;
  std::mt19937 rng_;
  std::size_t generation_;
};

}
