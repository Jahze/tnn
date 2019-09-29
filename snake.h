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
        policyGradient_(ActionCount), avgLength_(1000u),
        lastPolicyGradient_(ActionCount) {

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
      case 'G':
        shouldShowPolicyGradients_ = !shouldShowPolicyGradients_;
        break;
      case VK_LEFT:
        if (showingPolicyGradients_) {
          if (policyGradientFrame_ > 0ull)
            --policyGradientFrame_;
        }
        break;
      case VK_RIGHT:
        if (showingPolicyGradients_) {
          ++policyGradientFrame_;

          if (policyGradientFrame_ >= lastPolicyGradient_.FrameCount()) {
            showingPolicyGradients_ = false;
            lastOutputs_.clear();
          }
        }
        break;
      }
    });

    brain_.reset(new NeuralNet(gridSize_ * gridSize_, ActionCount, 1u, 100u));
    brain_->SetOptimiser(Optimiser::RMSProp);
    //brain_->SetOptimiser(Optimiser::AdamOptimiser);
    brain_->SetLearningRate(0.001);
    brain_->SetOutputLayerActivationType(ActivationType::Softmax);
    brain_->SetHiddenLayerActivationType(ActivationType::Sigmoid);
  }

protected:
  void StartImpl() override {
  }

  void PrintText(float x, float y, const std::string & text) {
    glColor3d(1.0, 1.0, 1.0);
    glRasterPos2f(x, y);
    glListBase(0);
    glCallLists(text.size(), GL_UNSIGNED_BYTE, text.c_str());
  }

  float StartPrintingText() {
    ScopedHDC hdc{context_.Handle()};

    ::wglUseFontBitmaps(hdc, 0, 255, 0);

    ::RECT rect;
    ::GetClientRect(context_.Handle(), &rect);

    ::TEXTMETRIC metrics;
    ::GetTextMetrics(hdc, &metrics);

    float height =
      static_cast<float>(metrics.tmHeight) /
      static_cast<float>(rect.right);

    height *= 2.0f;

    return height;
  }

  void PrintStats() {
    float height = StartPrintingText();

    double y = 1.0f - height;
    PrintText(-1.0f, y, "Length: " + std::to_string(snakePositions_.size()));

    y -= height;
    PrintText(-1.0f, y, "Idle moves: " + std::to_string(moves_));
  }

  std::string ActionName(size_t action) {
    switch (action) {
    case 0: return "Up";
    case 1: return "Down";
    case 2: return "Left";
    case 3: return "Right";
    default: return "ERROR!";
    }
  }

  void ShowPolicyGradients() {
    auto inputs = lastPolicyGradient_.Inputs(policyGradientFrame_);

    const double step = 2.0 / static_cast<double>(gridSize_);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);

    for (size_t x = 0ull; x < gridSize_; ++x) {
      for (size_t y = 0ull; y < gridSize_; ++y) {
        const double value = inputs[x * gridSize_ + y];
        const double r = value < 0.0 ? -value: 0.0;
        const double g = value > 0.0 ? value: 0.0;

        glColor3d(r, g, 0.0);
        glVertex2d(-1.0 + x * step,   -1.0 + y * step);
        glVertex2d(-1.0 + x+1 * step, -1.0 + y * step);
        glVertex2d(-1.0 + x+1 * step, -1.0 + y+1 * step);
        glVertex2d(-1.0 + x * step,   -1.0 + y+1 * step);
      }
    }

    glEnd();

    float height = StartPrintingText();
    float y = 1.0f - height;

    PrintText(-1.0f, y, "Frame: "
      + std::to_string(policyGradientFrame_ + 1ull)
      + "/" + std::to_string(lastPolicyGradient_.FrameCount()));

    y -= height * 2.0f;
    PrintText(-1.0f, y, "Up probability: " + std::to_string(lastOutputs_[policyGradientFrame_][0]));

    y -= height;
    PrintText(-1.0f, y, "Down probability: " + std::to_string(lastOutputs_[policyGradientFrame_][1]));

    y -= height;
    PrintText(-1.0f, y, "Left probability: " + std::to_string(lastOutputs_[policyGradientFrame_][2]));

    y -= height;
    PrintText(-1.0f, y, "Right probability: " + std::to_string(lastOutputs_[policyGradientFrame_][3]));

    y -= height;
    PrintText(-1.0f, y, "Sample: " + std::to_string(lastOutputs_[policyGradientFrame_][4]));

    auto action = lastPolicyGradient_.SelectedAction(policyGradientFrame_);

    y -= height * 2.0f;
    PrintText(-1.0f, y, "Selected action: " + ActionName(action));

    auto reward = lastPolicyGradient_.Reward(policyGradientFrame_);

    y -= height;
    PrintText(-1.0f, y, "Reward: " + std::to_string(reward));

    auto outputs = brain_->Process(inputs);

    y -= height * 2.0f;
    PrintText(-1.0f, y, "Next Up probability: " + std::to_string(outputs[0]));

    y -= height;
    PrintText(-1.0f, y, "Next Down probability: " + std::to_string(outputs[1]));

    y -= height;
    PrintText(-1.0f, y, "Next Left probability: " + std::to_string(outputs[2]));

    y -= height;
    PrintText(-1.0f, y, "Next Right probability: " + std::to_string(outputs[3]));

    context_.SwapBuffers();
  }

  void UpdateImpl(bool render, std::size_t ms) override {
    if (showingPolicyGradients_) {
      ShowPolicyGradients();
      return;
    }

    if (moves_++ > MaximumIdleMoves) {
      policyGradient_.UpdateLastReward(-1.0);
      Restart();

      if (showingPolicyGradients_)
        return;
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

    PrintStats();

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

    snakePositions_.push_back(next);

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
    if (shouldShowPolicyGradients_) {
      lastPolicyGradient_ = policyGradient_;
      showingPolicyGradients_ = true;
      policyGradientFrame_ = 0ull;
    }
    else {
      lastOutputs_.clear();
    }

    policyGradient_.Teach(*brain_.get());
    policyGradient_.Reset();

    avgLength_.AddDataPoint(snakePositions_.size());
    iteration_++;
    if (iteration_ % 1000u == 0u) {
      std::cout << "Iteration " << iteration_
        << " [length avg = " << avgLength_.Average() << " "
        << "max = " << avgLength_.Max() << "] "
        << iterationTimer_.ElapsedMicroseconds() << "\n";

      iterationTimer_.Reset();

      avgLength_.PrintHistogram();
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

    auto iter = std::rbegin(snakePositions_);
    const auto end = std::crend(snakePositions_);

    out[iter->x * gridSize_ + iter->y] = 1.0;

    ++iter;

    for ( ; iter != end; ++iter)
      out[iter->x * gridSize_ + iter->y] = 0.5;

    out[applePosition_.x * gridSize_ + applePosition_.y] = -1.0;

    return out;
  }

  void SampleBrain() {
    auto inputs = EncodeState();

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
    case 0: ChangeDirection(Direction::Up);     break;
    case 1: ChangeDirection(Direction::Down);   break;
    case 2: ChangeDirection(Direction::Left);   break;
    case 3: ChangeDirection(Direction::Right);  break;
    }

    policyGradient_.StoreIO(std::move(inputs), selectedAction);

    outputs.push_back(sample);

    lastOutputs_.push_back(std::move(outputs));
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

  const std::size_t ActionCount = 4u;
  const std::size_t MaximumIdleMoves = 50u;

  Position applePosition_;
  std::deque<Position> snakePositions_;
  bool growSnake_ = false;

  Direction snakeDirection_ = Direction::Right;
  Direction lastSnakeDirection_ = Direction::Right;

  std::default_random_engine rng_;

  std::unique_ptr<NeuralNet> brain_;
  PolicyGradient policyGradient_;
  Position lastPosition_;

  MovingAverage<double> avgLength_;
  Timer iterationTimer_;

  PolicyGradient lastPolicyGradient_;
  bool shouldShowPolicyGradients_ = true;
  bool showingPolicyGradients_ = false;
  size_t policyGradientFrame_ = 0ull;
  std::vector<std::vector<double>> lastOutputs_;
};

}
