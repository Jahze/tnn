#pragma once

#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <functional>
#include <memory>
#include <random>
#include <vector>
#include "macros.h"
#include "threadpool.h"

template<typename T>
class Aligned32ByteRAIIStorage {
public:
  Aligned32ByteRAIIStorage() : storage_(nullptr) {}

  Aligned32ByteRAIIStorage(std::size_t size) {
    size += (32 - ((size * sizeof(T)) % 32)) / sizeof(T);
    storage_ = new double [size];
  }

  ~Aligned32ByteRAIIStorage() { delete [] storage_; }

  Aligned32ByteRAIIStorage(const Aligned32ByteRAIIStorage<T> &) = delete;
  Aligned32ByteRAIIStorage& operator=(
    const Aligned32ByteRAIIStorage<T> &) = delete;

  Aligned32ByteRAIIStorage(Aligned32ByteRAIIStorage && rhs) {
    storage_ = rhs.storage_;
    rhs.storage_ = nullptr;
  }

  Aligned32ByteRAIIStorage& operator=(Aligned32ByteRAIIStorage && rhs) {
    storage_ = rhs.storage_;
    rhs.storage_ = nullptr;
  }

  T & operator[](std::size_t i) { return storage_[i]; }
  T operator[](std::size_t i) const { return storage_[i]; }
  T * Get() { return storage_; }
  const T * Get() const { return storage_; }

  void Reset(std::size_t size) {
    delete [] storage_;

    size += (32 - ((size * sizeof(T)) % 32)) / sizeof(T);
    storage_ = new double[size];
  }

private:
  T * storage_;
};

struct Neurone {
  const std::size_t size_;
  Aligned32ByteRAIIStorage<double> weights_;

  Neurone(std::size_t size) : size_(size), weights_(size+1) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<> distribution(
      0.0, sqrt(2.0/static_cast<double>(size)));

    for (auto i = 0u; i < size_; ++i)
      weights_[i] = distribution(generator);

    // Initialise bias to 0
    weights_[size_] = 0.0;
  }
};

struct NeuroneLayer {
  const std::size_t size_;
  // TODO: could be unique_ptr?
  std::vector<Neurone> neurones_;

  NeuroneLayer(std::size_t size, std::size_t neuroneSize)
    : size_(size) {
    for (auto i = 0u; i < size; ++i)
      neurones_.emplace_back(neuroneSize);
  }
};

class NeuralNet {
public:
  const static double kThresholdBias;
  const static double kActivationResponse;

  NeuralNet(std::size_t inputs,
    std::size_t outputs,
    std::size_t hiddenLayers,
    std::size_t neuronesPerHiddenLayer)
    : inputs_(inputs)
    , outputs_(outputs)
    , hiddenLayers_(hiddenLayers)
    , neuronesPerHiddenLayer_(neuronesPerHiddenLayer) {
    Create();
  }

  std::vector<double> Process(const std::vector<double> & inputs) {
    CHECK(inputs.size() == inputs_);

    Aligned32ByteRAIIStorage<double> * lastOutputs = buffers_.data();
    std::copy(inputs.cbegin(), inputs.cend(), lastOutputs->Get());

    Aligned32ByteRAIIStorage<double> * outputs = (lastOutputs + 1);

    std::size_t outputIndex = 0u;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      outputIndex = 0u;

      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        std::size_t inputIndex = 0;
        double activation = 0.0;

        const auto & neurone = layers_[i].neurones_[j];
        const std::size_t inputs = neurone.size_;

#if 0
        // activation += sum(weights[i] * inputs[i])
        const std::size_t batches = inputs / 4;
        for (std::size_t k = 0; k < batches; ++k) {
          __m256d weights = _mm256_load_pd(neurone.weights_.Get() + (k * 4));
          __m256d values = _mm256_load_pd(lastOutputs->Get() + (k * 4));
          __m256d result = _mm256_mul_pd(weights, values);
          __m256d accum = _mm256_hadd_pd(result, result);
          activation += accum.m256d_f64[0];
          activation += accum.m256d_f64[2];
          inputIndex += 4;
        }

        const std::size_t left = inputs % 4;
        for (std::size_t k = 0; k < left; ++k) {
          activation +=
            neurone.weights_[(batches * 4) + k] *
            lastOutputs->Get()[inputIndex++];
        }
#else
        for (std::size_t i = 0; i < inputs; ++i)
          activation += neurone.weights_[i] * lastOutputs->Get()[inputIndex++];
#endif

        activation += kThresholdBias * neurone.weights_[inputs];
        outputs->Get()[outputIndex++] =
          ActivationFunction(activation, kActivationResponse);
      }

      lastOutputs = outputs++;
    }

    return { lastOutputs->Get(), lastOutputs->Get() + outputIndex };
  }

  std::vector<double> GetWeights() const {
    std::vector<double> outputs;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        const auto & neurone = layers_[i].neurones_[j];
        const std::size_t inputs = neurone.size_;

        for (std::size_t k = 0, length = inputs; k < length; ++k)
          outputs.push_back(neurone.weights_[k]);

        outputs.push_back(neurone.weights_[inputs]);
      }
    }

    return outputs;
  }

  void SetWeights(const std::vector<double> & weights) {
    auto cursor = weights.begin();

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        auto & neurone = layers_[i].neurones_[j];
        const std::size_t inputs = neurone.size_;

        for (std::size_t k = 0, length = inputs; k < length; ++k)
          neurone.weights_[k] = *cursor++;

        neurone.weights_[inputs] = *cursor++;
      }
    }

    CHECK(cursor == weights.end());
  }

  double ActivationFunction(double activation, double response) const {
    return (1.0 / (1.0 + std::exp(-activation / response)));
  }

  double ActivationFunctionDerivative(double activation,
      double response) const {
    const double value = ActivationFunction(activation, response);
    return value * (1 - value);
  }

  void BackPropagation(const std::vector<double> & inputs,
      std::function<double(double,std::size_t)> lossFunctionDerivative) {

    Process(inputs);

    std::vector<double> weights;

    std::vector<double> last_dLoss_dOut;
    std::vector<double> next_dLoss_dOut;
    std::vector<double> last_dOut_dActivation;
    std::vector<double> next_dOut_dActivation;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      // Go backwards through the layers
      const std::size_t current = hiddenLayers_ - i;

      auto & layer = layers_[current];

      for (std::size_t j = 0; j < layer.size_; ++j) {
        const std::size_t neuroneIndex = layer.size_ - 1 - j;

        auto & neurone = layer.neurones_[neuroneIndex];
        const std::size_t inputs = neurone.size_;

        const double LearningRate = 0.5;
        const double out = buffers_[current + 1][neuroneIndex];

        // dLoss/dWeight = dLoss/dOut * dOut/dActivation * dActivation/dWeight

        const double dLoss_dOut = lossFunctionDerivative(out, neuroneIndex);
        next_dLoss_dOut.push_back(dLoss_dOut);

        // derivative of activation function
        const double dOut_dActivation = (out * (1.0 - out));
        next_dOut_dActivation.push_back(dOut_dActivation);

        // derivative of neuron output function multiplied in final calc
        const double dLoss_dActivation = dLoss_dOut * dOut_dActivation;

        // put weights in backwards

        // TODO: should we pass-through?
        //weights.push_back(neurone.weights_[inputs]);
        weights.push_back(neurone.weights_[inputs] -
          (LearningRate * dLoss_dActivation * kThresholdBias));

        for (std::size_t k = 0; k < inputs; ++k) {
          const std::size_t weightIndex = inputs - k - 1;

          const double divergence = LearningRate * dLoss_dActivation
            * buffers_[current][weightIndex];

          weights.push_back(neurone.weights_[weightIndex] - divergence);
        }
      }

      // dLoss/dOut = sum(dLoss_last/dActivation_last * dActivation/dOut_last) 
      // dLoss_last/dActivation_last =
      //          dLoss_last/dOut_last * dOut_last/dActivation_last
      // dActivation/dOut_last = weight

      last_dLoss_dOut = std::move(next_dLoss_dOut);
      std::reverse(last_dLoss_dOut.begin(), last_dLoss_dOut.end());
      next_dLoss_dOut.clear();

      last_dOut_dActivation = std::move(next_dOut_dActivation);
      std::reverse(last_dOut_dActivation.begin(), last_dOut_dActivation.end());
      next_dOut_dActivation.clear();

      lossFunctionDerivative =
        [this, &last_dLoss_dOut, &last_dOut_dActivation, current]
        (double, std::size_t index) {

          auto & layer = layers_[current];

          double value = 0.0;

          for (std::size_t i = 0; i < layer.size_; ++i) {
            value += last_dLoss_dOut[i] * last_dOut_dActivation[i]
              * layer.neurones_[i].weights_[index];
          }

          return value;
        };
    }

    std::reverse(weights.begin(), weights.end());
    SetWeights(weights);
  }

  void BackPropagationThreaded(const std::vector<double> & inputs,
      std::function<double(double,std::size_t)> lossFunctionDerivative) {

    Process(inputs);

    std::vector<double> weights;
    std::vector<std::vector<double>> layerWeights;

    std::vector<double> last_dLoss_dOut;
    std::vector<double> next_dLoss_dOut;
    std::vector<double> last_dOut_dActivation;
    std::vector<double> next_dOut_dActivation;

    ThreadPool * pool = GetCpuSizedThreadPool();

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      // Go backwards through the layers
      const std::size_t current = hiddenLayers_ - i;

      auto & layer = layers_[current];

      TaskList tasks(*pool);
      next_dLoss_dOut.resize(layer.size_);
      next_dOut_dActivation.resize(layer.size_);
      layerWeights.resize(layer.size_);

      for (std::size_t j = 0; j < layer.size_; ++j) {
        layerWeights[j].reserve(1 + layer.neurones_[layer.size_ - 1 - j].size_);

        tasks.AddTask([this, &layer, &layerWeights, &next_dLoss_dOut,
          &next_dOut_dActivation, &lossFunctionDerivative, j, current](){

          const std::size_t neuroneIndex = layer.size_ - 1 - j;

          auto & neurone = layer.neurones_[neuroneIndex];
          const std::size_t inputs = neurone.size_;

          const double LearningRate = 1.0;
          const double out = buffers_[current + 1][neuroneIndex];

          // dLoss/dWeight = dLoss/dOut * dOut/dActivation * dActivation/dWeight

          const double dLoss_dOut = lossFunctionDerivative(out, neuroneIndex);
          next_dLoss_dOut[j] = dLoss_dOut;

          // derivative of activation function
          const double dOut_dActivation = (out * (1.0 - out));
          next_dOut_dActivation[j] = dOut_dActivation;

          // derivative of neuron output function multiplied in final calc
          const double dLoss_dActivation = dLoss_dOut * dOut_dActivation;

          // put weights in backwards

          layerWeights[j].push_back(neurone.weights_[inputs] -
            (LearningRate * dLoss_dActivation * kThresholdBias));

          for (std::size_t k = 0; k < inputs; ++k) {
            const std::size_t weightIndex = inputs - k - 1;

            const double divergence = LearningRate * dLoss_dActivation
              * buffers_[current][weightIndex];

            layerWeights[j].push_back(neurone.weights_[weightIndex] - divergence);
          }
        });
      }

      tasks.Run();

      for (std::size_t j = 0; j < layer.size_; ++j) {
        weights.insert(weights.end(), layerWeights[j].begin(),
          layerWeights[j].end());
      }

      layerWeights.clear();

      last_dLoss_dOut = std::move(next_dLoss_dOut);
      std::reverse(last_dLoss_dOut.begin(), last_dLoss_dOut.end());
      next_dLoss_dOut.clear();

      last_dOut_dActivation = std::move(next_dOut_dActivation);
      std::reverse(last_dOut_dActivation.begin(), last_dOut_dActivation.end());
      next_dOut_dActivation.clear();

      lossFunctionDerivative =
        [this, &last_dLoss_dOut, &last_dOut_dActivation, current]
        (double, std::size_t index) {

          auto & layer = layers_[current];

          double value = 0.0;

          for (std::size_t i = 0; i < layer.size_; ++i) {
            value += last_dLoss_dOut[i] * last_dOut_dActivation[i]
              * layer.neurones_[i].weights_[index];
          }

          return value;
        };
    }

    std::reverse(weights.begin(), weights.end());
    SetWeights(weights);
  }

private:
  void Create() {
    CHECK(hiddenLayers_> 0);

    layers_.emplace_back(neuronesPerHiddenLayer_, inputs_);

    for (std::size_t i = 0, length = hiddenLayers_ - 1; i < length; ++i)
      layers_.emplace_back(neuronesPerHiddenLayer_, neuronesPerHiddenLayer_);

    layers_.emplace_back(outputs_, neuronesPerHiddenLayer_);

    const std::size_t BufferSize =
      std::max(inputs_, std::max(neuronesPerHiddenLayer_, outputs_));

    // extra layers for input and output
    const std::size_t bufferCount = hiddenLayers_ + 2;
    for (std::size_t i = 0; i < bufferCount; ++i)
      buffers_.emplace_back(BufferSize);
  }

private:
  std::size_t inputs_;
  std::size_t outputs_;
  std::size_t hiddenLayers_;
  std::size_t neuronesPerHiddenLayer_;

  std::vector<NeuroneLayer> layers_;

  std::vector<Aligned32ByteRAIIStorage<double>> buffers_;
};

struct Genome {
  std::vector<double> weights_;
  double fitness_;

  Genome(const std::vector<double> & weights, double fitness)
    : weights_(weights), fitness_(fitness) {}

  bool operator<(const Genome & rhs) const {
    return fitness_ < rhs.fitness_;
  }
};

class Generation {
public:
  Generation(const std::vector<Genome> & population)
    : population_(population), rng_(random_()) {}

  std::vector<Genome> GetPopulation() { return population_; }

  std::vector<std::vector<double>> Evolve();

  void SetMutationRate(double rate) { mutationRate_ = rate; }

private:
  const static double kCrossoverRate;
  const static double kMutationRate;
  const static double kMaxPeturbation;

  std::pair<std::vector<double>, std::vector<double>> Crossover(
    const Genome & mother,
    const Genome & father);

  std::pair<std::vector<double>, std::vector<double>> UniformCrossover(
    const Genome & mother,
    const Genome & father);

  void Mutate(std::vector<double> & genome);

  double RandomDouble() {
    std::uniform_real_distribution<double> distribution(0., 1.);
    return distribution(rng_);
  }

  double RandomDouble(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

  int RandomInt(int min, int max) {
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(rng_);
  }

  const Genome & SelectGenome(double totalFitness);

private:
  std::vector<Genome> population_;
  std::random_device random_;
  std::mt19937 rng_;
  double mutationRate_ = kMutationRate;
};

class Simulation {
public:
  Simulation(std::size_t msPerFrame, std::size_t msPerGenerationRender)
    : msPerFrame_(msPerFrame), msPerGenerationRender_(msPerGenerationRender) {}

  void Start() {
    lastTick_ = std::chrono::high_resolution_clock::now();
    timeSinceSpawn_ = std::chrono::milliseconds(0u);

    StartImpl();
  }

  void Update(bool render) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - lastTick_;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

    if (!render) ms = std::chrono::milliseconds(msPerFrame_ + 1);

    if (ms.count() > msPerFrame_) {
      UpdateImpl(render, ms.count());
      timeSinceSpawn_ += ms;

      lastTick_ = std::chrono::high_resolution_clock::now();
    }

    if (timeSinceSpawn_.count() > msPerGenerationRender_) {
      // Run it for longer than shown to get to the end
      const uint64_t extraTicks =
        (msPerGenerationRender_ * postRenderGenerations_) / msPerFrame_;
      for (uint64_t i = 0; i < extraTicks; ++i)
        UpdateImpl(false, msPerFrame_);

      Train();

      timeSinceSpawn_ = std::chrono::milliseconds(0u);
      lastTick_ = std::chrono::high_resolution_clock::now();
    }
  }

  void SetPostRenderGenerations(std::size_t generations) {
    postRenderGenerations_ = generations;
  }

protected:
  virtual void StartImpl() = 0;
  virtual void UpdateImpl(bool render, std::size_t ms) = 0;
  virtual void Train() = 0;

private:
  std::chrono::time_point<std::chrono::steady_clock> lastTick_;
  std::chrono::milliseconds timeSinceSpawn_;
  std::size_t msPerFrame_;
  std::size_t msPerGenerationRender_;
  std::size_t postRenderGenerations_ = 10u;
};
