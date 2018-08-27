#include <random>
#include "graphics.h"
#include "moving_average.h"
#include "simulation.h"

namespace snake {

class PolicyGradient {
public:
  PolicyGradient(std::size_t actionCount) : actionCount_(actionCount) {}

  void StoreIO(std::vector<double> && inputs, std::size_t action) {
    inputs_.push_back(std::move(inputs));
    actions_.push_back(action);
  }

  void StoreReward(double reward) {
    rewards_.push_back(reward);
  }

  void Reset() {
    inputs_.clear();
    actions_.clear();
    rewards_.clear();
  }

  void Teach(NeuralNet & net) {
    if (inputs_.empty()) return;

    auto outputs = net.Process(inputs_);

    const auto & rewards = DiscountedRewards();

    const std::size_t inputSize = inputs_.size();

    AlignedMatrix loss{inputSize, actionCount_};

    for (std::size_t i = 0u; i < inputSize; ++i) {
      for (std::size_t j = 0u; j < actionCount_; ++j) {
        double ideal = j == actions_[i] ? 1.0 : 0.0;
        loss[i][j] = (outputs[i][j] - ideal) * rewards[i];
        //loss[i][j] = rewards_[i];
      }
    }

    net.BackPropagationCrossEntropy(loss);

    std::cout << "Last action\n";
    std::cout << "{";
    std::cout << std::setprecision(2);

    for (std::size_t i = 0u; i < actionCount_; ++i)
      std::cout << outputs[inputSize - 1u][i] << ", ";

    std::cout << "}\n";
    std::cout << "Selected action = " << actions_[inputSize - 1u] << "\n";
    std::cout << "Reward = " << rewards[inputSize - 1u] << "\n";

    auto nextOutputs = net.Process(inputs_);

    std::cout << "Next action\n";
    std::cout << "{";
    std::cout << std::setprecision(2);

    for (std::size_t i = 0u; i < actionCount_; ++i)
      std::cout << nextOutputs[inputSize - 1u][i] << ", ";

    std::cout << "}\n---\n";

    if (rewards[inputSize - 1u] < 0.0 &&
        nextOutputs[inputSize - 1u][actions_[inputSize - 1u]] >
        outputs[inputSize - 1u][actions_[inputSize - 1u]]) {
      std::cout << "HOW?\n";
    }
  }

private:
  std::vector<double> DiscountedRewards() const {
    const std::size_t inputSize = inputs_.size();

    std::vector<double> out(inputSize);

    const double Gamma = 0.95;
    double reward = 0.0;
    double mean = 0.0;

    for (std::size_t i = inputSize; i > 0u; --i) {
      reward = reward * Gamma + rewards_[i - 1u];
      out[i - 1u] = reward;
      mean += reward;
    }

    mean /= inputSize;

    double stddev = 0.0;

    for (std::size_t i = 0u; i < inputSize; ++i)
      stddev += std::pow(out[i] - mean, 2.);

    stddev = std::sqrt(stddev / inputSize);

    for (std::size_t i = 0u; i < inputSize; ++i) {
      out[i] -= mean;
      out[i] /= stddev;
    }

    return out;
  }

private:
  const std::size_t actionCount_;
  std::vector<std::vector<double>> inputs_;
  std::vector<std::size_t> actions_;
  std::vector<double> rewards_;
};

class Snake : public ::SimpleSimulation {
public:
  Snake(std::size_t msPerFrame, OpenGLContext & context, std::size_t gridSize)
      : SimpleSimulation(msPerFrame), context_(context), gridSize_(gridSize),
        policyGradient_(ActionCount), avgLength_(10u) {

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
      }
    });

    // Pass last two frames so it can detect movement

    //brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 10u));
    brain_.reset(new NeuralNet(gridSize_ * gridSize_ * 2, ActionCount, 1u, 100u));
    brain_->SetLearningRate(0.01);
    brain_->SetOutputLayerActivationType(ActivationType::Softmax);
    brain_->SetHiddenLayerActivationType(ActivationType::ReLu);
  }

protected:
  void StartImpl() override {
  }

  void UpdateImpl(bool render, std::size_t ms) override {
    if (!render) return;

    SampleBrain();

    MoveSnake();

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

    //const double MaxDistance = std::sqrt(std::pow(gridSize_ - 1u, 2.0) + std::pow(gridSize_ - 1u, 2.0));

    //const double xdist = int64_t(next.x) - int64_t(applePosition_.x);
    //const double ydist = int64_t(next.y) - int64_t(applePosition_.y);
    //const double dist = std::sqrt(xdist*xdist + ydist*ydist);
    //const double scaled = 1.0 - dist / MaxDistance;
    ////std::cout << "distance " << scaled << "\n";
    //policyGradient_.StoreReward(scaled);

    if (growSnake_) {
      PlaceApple();
      policyGradient_.StoreReward(1.);
    }
    else {
      //policyGradient_.StoreReward(0.1);
      policyGradient_.StoreReward(0.0);
      //policyGradient_.StoreReward(-0.01);
    }

    snakePositions_.push_back(next);

    lastSnakeDirection_ = snakeDirection_;
  }

  void Dead() {
    policyGradient_.StoreReward(-1.);
    policyGradient_.Teach(*brain_.get());
    policyGradient_.Reset();

    avgLength_.AddDataPoint(snakePositions_.size());
    iteration_++;
    std::cout << "Iteration " << iteration_
      << " [length avg = " << avgLength_.Average() << "]\n";

    Reset();
  }

  void Reset() {
    std::size_t mid = gridSize_ / 2;

    snakePositions_.clear();
    snakePositions_.push_back({mid - 1, mid});
    snakePositions_.push_back({mid, mid});

    snakeDirection_ = Direction::Right;
    lastSnakeDirection_ = Direction::Right;

    PlaceApple();

    lastState_ = EncodeState();
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

    // XXX below doesn't run
    //auto state = EncodeState();

    //const std::size_t StateSize = state.size();

    //std::vector<double> out(StateSize);

    //// TODO: normalize
    //for (std::size_t i = 0u; i < StateSize; ++i)
    //  out[i] = state[i] - lastState_[i];

    //// TODO: not sure if i should do this
    //// the problem is that using a frame difference detect movement
    //// but not unchanging position of apple
    //out[applePosition_.x * gridSize_ + applePosition_.y] = 0.5;

    //lastState_ = std::move(state);

    //return out;
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

  const std::size_t ActionCount = 5u;

  struct Position {
    std::size_t x;
    std::size_t y;
  };

  Position applePosition_;
  std::deque<Position> snakePositions_;
  bool growSnake_ = false;

  Direction snakeDirection_ = Direction::Right;
  Direction lastSnakeDirection_ = Direction::Right;

  std::default_random_engine rng_;

  std::unique_ptr<NeuralNet> brain_;
  PolicyGradient policyGradient_;
  std::vector<double> lastState_;

  MovingAverage<double> avgLength_;
};

}
