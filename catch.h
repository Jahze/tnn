#pragma once

#include <random>
#include "graphics.h"
#include "moving_average.h"
#include "policy_gradient.h"
#include "simulation.h"
#include "timer.h"

namespace catcher {

class Catch : public ::SimpleSimulation {
private:
  struct Position {
    std::size_t x;
    std::size_t y;
  };

public:
  Catch(std::size_t msPerFrame, OpenGLContext & context, std::size_t gridSize)
      : SimpleSimulation(msPerFrame), context_(context), gridSize_(gridSize),
        policyGradient_(ActionCount), avgScore_(1000u) {

    std::random_device r;
    rng_.seed(r());

    Reset();

    //context.AddKeyListener([this](int key) {
    //  switch (key) {
    //  case VK_LEFT:
    //    Move(Direction::Left);
    //    break;
    //  case VK_RIGHT:
    //    Move(Direction::Right);
    //    break;
    //  }
    //});

    context.AddKeyListener([this](int key) {
      static bool fast = true;
      switch (key) {
      case VK_SPACE:
        SetMsPerFrame(fast?250u:10u);
        fast = !fast;
        break;
      case 'S':
        brain_->SerializeWeights("catch-brain.txt");
        break;
      case 'L':
        brain_->DeserializeWeights("catch-brain.txt");
        break;
      }
    });

    // Pass last two frames so it can detect movement

    //brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 10u));
    brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 100u));
    //brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 300u));
    brain_->SetOptimiser(Optimiser::AdamOptimiser);
    //brain_->SetOptimiser(Optimiser::RMSProp);
    brain_->SetLearningRate(0.001);
    brain_->SetOutputLayerActivationType(ActivationType::Softmax);
    brain_->SetHiddenLayerActivationType(ActivationType::ReLu);
  }

protected:
  void StartImpl() override {
  }

  void UpdateImpl(bool render, std::size_t ms) override {
    SampleBrain();

    MoveApples();

    if (!render) return;

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);

      const double step = 2.0 / static_cast<double>(gridSize_);

      const auto & pos = catcherPosition_;
      glColor3d(1.0, 1.0, 1.0);
      glVertex2d(-1.0 + pos.x * step,      -1.0 + pos.y * step);
      glVertex2d(-1.0 + (pos.x+1) * step,  -1.0 + pos.y * step);
      glVertex2d(-1.0 + (pos.x+1) * step,  -1.0 + (pos.y+1) * step);
      glVertex2d(-1.0 + pos.x * step,      -1.0 + (pos.y+1) * step);

      const auto & apple = applePosition_;
      glColor3d(0.8, 0.4, 0.4);
      glVertex2d(-1.0 + apple.x * step,      -1.0 + apple.y * step);
      glVertex2d(-1.0 + (apple.x+1) * step,  -1.0 + apple.y * step);
      glVertex2d(-1.0 + (apple.x+1) * step,  -1.0 + (apple.y+1) * step);
      glVertex2d(-1.0 + apple.x * step,      -1.0 + (apple.y+1) * step);

    glEnd();

    context_.SwapBuffers();
  }

  void MoveApples() {
    if (applePosition_.y == 0) {
      policyGradient_.StoreReward(-1.0);

      //if (score_ == 0u) {
      //  Restart();
      //}
      //else {
      //  score_--;
      //  PlaceApple();
      //}

      Restart();

      return;
    }

    applePosition_.y--;

    if (HitsApple()) {
      policyGradient_.StoreReward(1.0);

      score_++;
      //std::cout << "Score: " << score_ << "\n";
      PlaceApple();
      maxScore_ = std::max(score_, maxScore_);

      //if(score_ == 800u) {
      //  extern bool g_render;
      //  g_render = true;
      //  SetMsPerFrame(250u);
      //}

      if (score_ > 100u)
        Restart();
    }
    else {
      policyGradient_.StoreReward(0.0);
    }
  }

  void Restart() {
    policyGradient_.Teach(*brain_.get());
    policyGradient_.Reset();

    avgScore_.AddDataPoint(maxScore_);
    iteration_++;
    if (iteration_ % 1000u == 0u) {
      std::cout << "Iteration " << iteration_
        << " [score avg = " << avgScore_.Average() << "] "
        << iterationTimer_.ElapsedMicroseconds() << "\n";

      iterationTimer_.Reset();

      if (maxLength_ == 0.0) {
        maxLength_ = avgScore_.Average();
      }
      else {
        double avg = avgScore_.Average();
        if (maxLength_ - avg >= 2.0) {
          std::cout << "Dropping from max\n";
        }
        maxLength_ = std::max(maxLength_, avg);
      }
    }

    Reset();
  }

  void Reset() {
    std::size_t mid = gridSize_ / 2;

    catcherPosition_ = {mid, 0u};

    score_ = 0u;
    maxScore_ = 0u;

    PlaceApple();

    lastState_ = EncodeState();
  }

  void PlaceApple() {
    std::uniform_int_distribution<std::size_t> distrib(0, gridSize_ - 1u);

    applePosition_.x = distrib(rng_);
    applePosition_.y = gridSize_ - 1u;
  }

  bool HitsApple() const {
    return applePosition_.x == catcherPosition_.x
      && applePosition_.y == catcherPosition_.y;
  }

  std::vector<double> EncodeState() {
    std::vector<double> out(gridSize_ * gridSize_, 0.0);

    out[catcherPosition_.x * gridSize_ + catcherPosition_.y] = 1.0;

    out[applePosition_.x * gridSize_ + applePosition_.y] = 0.5;

    return out;
  }

  std::vector<double> GenerateBrainInputs() {
    auto state = EncodeState();

    const std::size_t StateSize = state.size();

    std::vector<double> out(StateSize * 2);
    std::copy(lastState_.begin(), lastState_.end(), out.begin());
    std::copy(state.begin(), state.end(), out.begin() + StateSize);

    lastState_ = std::move(state);
    return out;
  }

  void SampleBrain() {
    auto inputs = GenerateBrainInputs();

    auto outputs = brain_->Process(inputs);

    std::uniform_real_distribution<double> distrib(0., 1.);
    double sample = distrib(rng_);

    double accum = 0.0;
    std::size_t selectedAction = -1;

    for (std::size_t i = 0u; i < ActionCount; ++i) {
      accum += outputs[i];
      if (accum >= sample) {
        selectedAction = i;
        break;
      }
    }

    switch (selectedAction) {
    case 1: Move(Direction::Left);   break;
    case 2: Move(Direction::Right);  break;
    }

    policyGradient_.StoreIO(std::move(inputs), selectedAction);
  }

  enum class Direction {
    Left,
    Right
  };

  void Move(Direction direction) {
    switch (direction) {
    case Direction::Left:
      if (catcherPosition_.x > 0u)
        catcherPosition_.x--;
      break;
    case Direction::Right:
      if (catcherPosition_.x < (gridSize_ - 1u))
        catcherPosition_.x++;
      break;
    }
  }

private:
  OpenGLContext & context_;
  const std::size_t gridSize_;
  std::size_t iteration_ = 0u;
  std::size_t score_ = 0u;
  std::size_t maxScore_ = 0u;

  const std::size_t ActionCount = 3u;

  Position applePosition_;
  Position catcherPosition_;

  std::default_random_engine rng_;

  std::unique_ptr<NeuralNet> brain_;
  PolicyGradient policyGradient_;
  std::vector<double> lastState_;
  Position lastPosition_;
  MovingAverage<double> avgScore_;
  double maxLength_ = 0.0;

  Timer iterationTimer_;
};

}
