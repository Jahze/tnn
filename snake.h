#pragma once

#include <random>
#include "graphics.h"
#include "moving_average.h"
#include "policy_gradient.h"
#include "simulation.h"

namespace snake {

class Snake : public ::SimpleSimulation {
private:
  struct Position {
    std::size_t x;
    std::size_t y;
  };

public:
  Snake(std::size_t msPerFrame, OpenGLContext & context, std::size_t gridSize)
      : SimpleSimulation(msPerFrame), context_(context), gridSize_(gridSize),
        policyGradient_(ActionCount), avgLength_(1000u) {

    std::random_device r;
    rng_.seed(r());

    Reset();

    //context.AddKeyListener([this](int key) {
    //  switch (key) {
    //  case VK_UP:
    //    ChangeDirection(Direction::Up);
    //    break;
    //  case VK_DOWN:
    //    ChangeDirection(Direction::Down);
    //    break;
    //  case VK_LEFT:
    //    ChangeDirection(Direction::Left);
    //    break;
    //  case VK_RIGHT:
    //    ChangeDirection(Direction::Right);
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
        brain_->SerializeWeights("snake-brain.txt");
        break;
      case 'L':
        brain_->DeserializeWeights("snake-brain.txt");
        break;
      }
    });

    // Pass last two frames so it can detect movement

    //brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 10u));
    brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 100u));
    //brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 300u));
    brain_->SetOptimiser(Optimiser::Momentum);
    brain_->SetLearningRate(0.001);
    brain_->SetOutputLayerActivationType(ActivationType::Softmax);
    brain_->SetHiddenLayerActivationType(ActivationType::ReLu);
  }

protected:
  void StartImpl() override {
  }

  void UpdateImpl(bool render, std::size_t ms) override {
    if (moves_++ > MaximumIdleMoves) {
      policyGradient_.UpdateLastReward(-1.0);
      Restart();
    }

    SampleBrain();

    MoveSnake();

    if (!render) return;

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);

      const double colourStep = 1.0 / static_cast<double>(gridSize_);
      const double step = 2.0 / static_cast<double>(gridSize_);

      for (auto && pos : snakePositions_) {
        glColor3d(pos.x * colourStep, pos.y * colourStep, 0.0);
        glVertex2d(-1.0 + pos.x * step,      -1.0 + pos.y * step);
        glVertex2d(-1.0 + (pos.x+1) * step,  -1.0 + pos.y * step);
        glVertex2d(-1.0 + (pos.x+1) * step,  -1.0 + (pos.y+1) * step);
        glVertex2d(-1.0 + pos.x * step,      -1.0 + (pos.y+1) * step);
      }

      const auto & apple = applePosition_;
      glColor3d(1.0, 1.0, 1.0);
      glVertex2d(-1.0 + apple.x * step,      -1.0 + apple.y * step);
      glVertex2d(-1.0 + (apple.x+1) * step,  -1.0 + apple.y * step);
      glVertex2d(-1.0 + (apple.x+1) * step,  -1.0 + (apple.y+1) * step);
      glVertex2d(-1.0 + apple.x * step,      -1.0 + (apple.y+1) * step);

    glEnd();

    context_.SwapBuffers();
  }

  void MoveSnake() {
    Position & current = snakePositions_.back();
    Position next = current;

    switch (snakeDirection_) {
    case Direction::Down:
      if (next.y == 0) {
        Dead();
        return;
      }
      next.y--;
      break;

    case Direction::Up:
      next.y++;
      if (next.y >= gridSize_) {
        Dead();
        return;
      }
      break;

    case Direction::Left:
      if (next.x == 0) {
        Dead();
        return;
      }
      next.x--;
      break;

    case Direction::Right:
      next.x++;
      if (next.x >= gridSize_) {
        Dead();
        return;
      }
      break;
    }

    if (HitsSnake(next.x, next.y)) {
      Dead();
      return;
    }

    if (!growSnake_)
      snakePositions_.pop_front();

    growSnake_ = HitsApple(next.x, next.y);

    if (growSnake_) {
      PlaceApple();
      policyGradient_.StoreReward(1.0);
      moves_ = 0u; // reset idle moves
    }
    else {
      //const double lastDistance = ScaledDistanceToApple(snakePositions_.back());
      //const double nextDistance = ScaledDistanceToApple(next);

      //policyGradient_.StoreReward(lastDistance > nextDistance ? 0.05 : 0.0);

      policyGradient_.StoreReward(0.0);
    }

    snakePositions_.push_back(next);

    lastSnakeDirection_ = snakeDirection_;
  }

  double ScaledDistanceToApple(Position p1) {
    const double MaxDistance = std::sqrt(std::pow(gridSize_ - 1u, 2.0)
        + std::pow(gridSize_ - 1u, 2.0));

    const double xdist = int64_t(p1.x) - int64_t(applePosition_.x);
    const double ydist = int64_t(p1.y) - int64_t(applePosition_.y);
    const double dist = std::sqrt(xdist*xdist + ydist*ydist);
    return dist / MaxDistance;
  }

  void Dead() {
    policyGradient_.StoreReward(-1.);
    Restart();
  }

  void Restart() {
    policyGradient_.Teach(*brain_.get());
    policyGradient_.Reset();

    avgLength_.AddDataPoint(snakePositions_.size());
    iteration_++;
    if (iteration_ % 1000u == 0u) {
      std::cout << "Iteration " << iteration_
        << " [length avg = " << avgLength_.Average() << "] "
        << iterationTimer_.ElapsedMicroseconds() << "\n";

      iterationTimer_.Reset();

      if (maxLength_ == 0.0) {
        maxLength_ = avgLength_.Average();
      }
      else {
        double avg = avgLength_.Average();
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

    snakePositions_.clear();
    snakePositions_.push_back({mid - 1, mid});
    snakePositions_.push_back({mid, mid});

    snakeDirection_ = Direction::Right;
    lastSnakeDirection_ = Direction::Right;

    moves_ = 0u;

    PlaceApple();

    lastState_ = EncodeState();
    lastPosition_ = snakePositions_.back();
  }

  void PlaceApple() {
    // no space for an apple
    if (snakePositions_.size() == gridSize_ * gridSize_)
      return;

    std::uniform_int_distribution<std::size_t> distrib(0, gridSize_ - 1u);

    while (true) {
      std::size_t x = distrib(rng_);
      std::size_t y = distrib(rng_);

      if (HitsSnake(x, y))
        continue;

      applePosition_.x = x;
      applePosition_.y = y;
      break;
    }
  }

  bool HitsSnake(std::size_t x, std::size_t y) const {
    for (auto && position : snakePositions_) {
      if (position.x == x && position.y == y)
        return true;
    }

    return false;
  }

  bool HitsApple(std::size_t x, std::size_t y) const {
    return applePosition_.x == x && applePosition_.y == y;
  }

  std::vector<double> EncodeState() {
    std::vector<double> out(gridSize_ * gridSize_, 0.0);

    for (auto && position : snakePositions_) {
      out[position.x * gridSize_ + position.y] = 1.0;
    }

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
    case 1: ChangeDirection(Direction::Up);     break;
    case 2: ChangeDirection(Direction::Down);   break;
    case 3: ChangeDirection(Direction::Left);   break;
    case 4: ChangeDirection(Direction::Right);  break;
    }

    policyGradient_.StoreIO(std::move(inputs), selectedAction);

#if 0
    std::cout << std::setprecision(2);
    std::cout << "Actions:\n";
    std::cout << outputs[0] << " = do nothing\n";
    std::cout << outputs[1] << " = up\n";
    std::cout << outputs[2] << " = down\n";
    std::cout << outputs[3] << " = left\n";
    std::cout << outputs[4] << " = right\n";
    std::cout << "Total = " <<
      outputs[0] + outputs[1] + outputs[2] + outputs[3] + outputs[4] << std::endl;

    std::cout << "Selected action = " << selectedAction << std::endl;
#endif
  }

  enum class Direction {
    Up,
    Down,
    Left,
    Right
  };

  void ChangeDirection(Direction direction) {
    switch (direction) {
    case Direction::Up:
      if (lastSnakeDirection_ != Direction::Down)
        snakeDirection_ = Direction::Up;
      break;
    case Direction::Down:
      if (lastSnakeDirection_ != Direction::Up)
        snakeDirection_ = Direction::Down;
      break;
    case Direction::Left:
      if (lastSnakeDirection_ != Direction::Right)
        snakeDirection_ = Direction::Left;
      break;
    case Direction::Right:
      if (lastSnakeDirection_ != Direction::Left)
        snakeDirection_ = Direction::Right;
      break;
    }
  }

private:
  OpenGLContext & context_;
  const std::size_t gridSize_;
  std::size_t iteration_ = 0u;
  std::size_t moves_ = 0u;

  const std::size_t ActionCount = 5u;
  const std::size_t MaximumIdleMoves = 100u;

  Position applePosition_;
  std::deque<Position> snakePositions_;
  bool growSnake_ = false;

  Direction snakeDirection_ = Direction::Right;
  Direction lastSnakeDirection_ = Direction::Right;

  std::default_random_engine rng_;

  std::unique_ptr<NeuralNet> brain_;
  PolicyGradient policyGradient_;
  std::vector<double> lastState_;
  Position lastPosition_;
  MovingAverage<double> avgLength_;
  double maxLength_ = 0.0;

  Timer iterationTimer_;
};

}