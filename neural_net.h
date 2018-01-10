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
  Aligned32ByteRAIIStorage() : storage_(nullptr), size_(0) {}

  Aligned32ByteRAIIStorage(std::size_t size) {
    size_ = size + (32 - ((size * sizeof(T)) % 32)) / sizeof(T);
    storage_ = new double [size_];
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
    // Don't resize unless we need more space
    if (size < size_) return;

    delete [] storage_;

    size_ = size + (32 - ((size * sizeof(T)) % 32)) / sizeof(T);
    storage_ = new double[size_];
  }

private:
  T * storage_;
  std::size_t size_;
};

enum class ActivationType {
  Sigmoid,
  Tanh,
  ReLu,
};

double ActivationFunction(ActivationType type, double activation);
double ActivationFunctionDerivative(ActivationType type, double value);

struct Neurone {
  double * ptr_;

  Neurone(double * ptr) : ptr_(ptr) {}

  double Weights(std::size_t i) const {
    return ptr_[i];
  }

  double & Weights(std::size_t i) {
    return ptr_[i];
  }
};

struct NeuroneLayer {
  const std::size_t size_;
  const std::size_t neuroneSize_;
  const std::size_t weightsPerNeurone_;
  Aligned32ByteRAIIStorage<double> weights_;
  Aligned32ByteRAIIStorage<double> outputs_;
  ActivationType activationType_ = ActivationType::Sigmoid;

  NeuroneLayer(std::size_t size, std::size_t neuroneSize)
    : size_(size), neuroneSize_(neuroneSize)
    , weightsPerNeurone_(neuroneSize_ + 1) {
    // One extra for bias
    const std::size_t numWeights = size_ * weightsPerNeurone_;

    outputs_.Reset(size_ + 1);
    weights_.Reset(numWeights);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<> distribution(
      0.0, sqrt(2.0 / static_cast<double>(size)));

    for (std::size_t i = 0; i < size_; ++i) {
      const std::size_t base = i * weightsPerNeurone_;

      for (std::size_t j = 0; j < neuroneSize_; ++j)
        weights_[base + j] = distribution(generator);

      // Initialise bias to 0
      weights_[base + neuroneSize] = 0.0;
    }
  }

  Neurone Neurones(std::size_t i) {
    return weights_.Get() + (i * weightsPerNeurone_);
  }

  double Weights(std::size_t i, std::size_t j) const {
    const double * ptr = weights_.Get() + (i * weightsPerNeurone_);
    return ptr[j];
  }

  const static double kThresholdBias;

  void Process(Aligned32ByteRAIIStorage<double> & inputs) {
    std::size_t weightsIndex = 0;

    for (std::size_t i = 0u; i < size_; ++i) {
      outputs_[i] = 0.0;

      for (std::size_t j = 0u; j < weightsPerNeurone_; ++j)
        outputs_[i] += inputs[j] * weights_[weightsIndex++];

      outputs_[i] = ActivationFunction(activationType_, outputs_[i]);
    }

    outputs_[size_] = kThresholdBias;
  }

  void ProcessThreaded(Aligned32ByteRAIIStorage<double> & inputs) {
    ThreadPool * pool = GetCpuSizedThreadPool();

    BatchTasks tasks(*pool);
    tasks.CreateBatches(size_,
      [this, &inputs](std::size_t start, std::size_t end) {

      for (std::size_t i = start; i < end; ++i) {
        outputs_[i] = 0.0;

        std::size_t weightsIndex = i * weightsPerNeurone_;

        for (std::size_t j = 0u; j < weightsPerNeurone_; ++j)
          outputs_[i] += inputs[j] * weights_[weightsIndex++];

        outputs_[i] = ActivationFunction(activationType_, outputs_[i]);
      }
    });

    tasks.Run();

    outputs_[size_] = kThresholdBias;
  }
};

class NeuralNet {
public:
  const static double kThresholdBias;

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

    Aligned32ByteRAIIStorage<double> inputsStorage(inputs.size() + 1);
    std::copy(inputs.cbegin(), inputs.cend(), inputsStorage.Get());
    inputsStorage[inputs.size()] = kThresholdBias;

    Aligned32ByteRAIIStorage<double> * in = &inputsStorage;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      layers_[i].Process(*in);

      in = &layers_[i].outputs_;
    }

    return { in->Get(), in->Get() + layers_.back().size_ };
  }

  std::vector<double> ProcessThreaded(const std::vector<double> & inputs) {
    CHECK(inputs.size() == inputs_);

    Aligned32ByteRAIIStorage<double> inputsStorage(inputs.size() + 1);
    std::copy(inputs.cbegin(), inputs.cend(), inputsStorage.Get());
    inputsStorage[inputs.size()] = kThresholdBias;

    Aligned32ByteRAIIStorage<double> * in = &inputsStorage;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      layers_[i].ProcessThreaded(*in);

      in = &layers_[i].outputs_;
    }

    return { in->Get(), in->Get() + layers_.back().size_ };
  }

  std::vector<double> GetWeights() const {
    std::vector<double> outputs;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      for (std::size_t j = 0; j < layers_[i].size_; ++j) {
        const std::size_t inputs = layers_[i].neuroneSize_;

        for (std::size_t k = 0, length = inputs; k < length; ++k)
          outputs.push_back(layers_[i].Weights(j, k));

        outputs.push_back(layers_[i].Weights(j, inputs));
      }
    }

    return outputs;
  }

  void SetWeights(const std::vector<double> & weights) {
    auto cursor = weights.data();
    auto layer = layers_.data();

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      for (std::size_t j = 0, length = layer->size_; j < length; ++j) {
        auto & neurone = layer->Neurones(j);
        const std::size_t inputs = layer->neuroneSize_;

        std::memcpy(neurone.ptr_, cursor,
          sizeof(double) * (inputs + 1));

        cursor += (inputs + 1);
      }
      ++layer;
    }

    CHECK(cursor == weights.data() + weights.size());
  }

  void SetLearningRate(double rate) {
    learningRate_ = rate;
  }

  void SetHiddenLayerActivationType(ActivationType type) {
    // Tanh needs a lower learning rate generally
    for (std::size_t i = 0; i < hiddenLayers_; ++i)
      layers_[i].activationType_ = type;
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

        auto & neurone = layer.Neurones(neuroneIndex);
        const std::size_t inputs = layer.neuroneSize_;

        const double LearningRate = learningRate_;
        const double out = buffers_[current + 1][neuroneIndex];

        // dLoss/dWeight = dLoss/dOut * dOut/dActivation * dActivation/dWeight

        const double dLoss_dOut = lossFunctionDerivative(out, neuroneIndex);
        next_dLoss_dOut.push_back(dLoss_dOut);

        // derivative of activation function
        const double dOut_dActivation =
          ActivationFunctionDerivative(layer.activationType_, out);

        next_dOut_dActivation.push_back(dOut_dActivation);

        // derivative of neuron output function multiplied in final calc
        const double dLoss_dActivation = dLoss_dOut * dOut_dActivation;

        // put weights in backwards
        weights.push_back(neurone.Weights(inputs) -
          (LearningRate * dLoss_dActivation * kThresholdBias));

        for (std::size_t k = 0; k < inputs; ++k) {
          const std::size_t weightIndex = inputs - k - 1;

          const double divergence = LearningRate * dLoss_dActivation
            * buffers_[current][weightIndex];

          weights.push_back(neurone.Weights(weightIndex) - divergence);
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
              * layer.Neurones(i).Weights(index);
          }

          return value;
        };
    }

    std::reverse(weights.begin(), weights.end());
    SetWeights(weights);
  }

  void BackPropagationThreaded(const std::vector<double> & inputs,
      std::function<double(double,std::size_t)> lossFunctionDerivative) {

    std::copy(inputs.cbegin(), inputs.cend(), buffers_[0].Get());

    const auto & outputs = ProcessThreaded(inputs);

    const std::size_t length = outputs.size();

    Aligned32ByteRAIIStorage<double> loss(length);

    for (std::size_t i = 0u; i < length; ++i)
      loss[i] = lossFunctionDerivative(outputs[i], i);

    BackPropagationThreaded(loss);
  }

  void BackPropagationThreaded(const Aligned32ByteRAIIStorage<double> & lossIn) {
    Aligned32ByteRAIIStorage<double> next_dLoss_dActivation;

    Aligned32ByteRAIIStorage<double> lastLoss;

    const double * loss = lossIn.Get();

    ThreadPool * pool = GetCpuSizedThreadPool();

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      // Go backwards through the layers
      const std::size_t current = hiddenLayers_ - i;

      auto & layer = layers_[current];

      next_dLoss_dActivation.Reset(layer.size_);

      //BatchTasks tasks(*pool);
      //tasks.CreateBatches(layer.size_, [this, &layer,
      //  &next_dLoss_dActivation, &loss, current]
      //  (std::size_t start, std::size_t end) {

      //  for (std::size_t k = start; k < end; ++k) {
      //    const double out = layers_[current].outputs_[k];

      //    // dLoss/dWeight = dLoss/dOut * dOut/dActivation * dActivation/dWeight

      //    const double dLoss_dOut = loss[k];

      //    // derivative of activation function
      //    const double dOut_dActivation =
      //      ActivationFunctionDerivative(layer.activationType_, out);

      //    // derivative of neuron output function multiplied in final calc
      //    const double dLoss_dActivation = dLoss_dOut * dOut_dActivation;

      //    next_dLoss_dActivation[k] = dLoss_dActivation;
      //  }
      //});

      //tasks.Run();

      for (std::size_t k = 0u; k < layer.size_; ++k) {
        const double out = layers_[current].outputs_[k];

        // derivative of activation function
        next_dLoss_dActivation[k] =
          ActivationFunctionDerivative(layer.activationType_, out);
      }

      const std::size_t batches = layer.size_ / 4;
      for (std::size_t k = 0; k < batches; ++k) {
        __m256d dLoss_dOut = _mm256_load_pd(loss + (k * 4));
        __m256d dOut_dActivation = _mm256_load_pd(next_dLoss_dActivation.Get() + (k * 4));
        __m256d result = _mm256_mul_pd(dLoss_dOut, dOut_dActivation);
        std::memcpy(next_dLoss_dActivation.Get() + (k * 4),
          result.m256d_f64, sizeof(double) * 4);
      }

      const std::size_t left = layer.size_ % 4;
      for (std::size_t k = 0; k < left; ++k) {
        next_dLoss_dActivation[(batches * 4) + k] =
          loss[(batches * 4) + k] *
          next_dLoss_dActivation[(batches * 4) + k];
      }

      if (current > 0u) {
        const std::size_t layerSize = layers_[current].size_;
        const std::size_t prevLayerSize = layers_[current - 1].size_;

        lastLoss.Reset(prevLayerSize);
        std::memset(lastLoss.Get(), 0, prevLayerSize * sizeof(double));

        for (std::size_t i = 0u; i < prevLayerSize; ++i) {
          for (std::size_t j = 0u; j < layerSize; ++j) {
            lastLoss[i] += next_dLoss_dActivation[j]
              * layers_[current].Neurones(j).Weights(i);
          }
        }

        loss = lastLoss.Get();
      }

      BatchTasks tasks2(*pool);
      tasks2.CreateBatches(layer.size_, [this, &layer,
        &next_dLoss_dActivation, current]
        (std::size_t start, std::size_t end) {

        for (std::size_t k = start; k < end; ++k) {
          auto & neurone = layer.Neurones(k);
          const std::size_t inputs = layer.neuroneSize_;

          const double LearningRate = learningRate_;

          for (std::size_t l = 0; l < inputs; ++l) {
            const double divergence = LearningRate * next_dLoss_dActivation[k]
              * (current == 0 ? buffers_[current][l] : layers_[current-1].outputs_[l]);

            neurone.Weights(l) -= divergence;
          }

          neurone.Weights(inputs) -=
            LearningRate * next_dLoss_dActivation[k] * kThresholdBias;
        }
      });

      tasks2.Run();
    }
  }

  //std::function<double(double, std::size_t)> LastLossFunction(
  //  const std::vector<double> & inputs,
  //  std::function<double(double,std::size_t)> lossFunctionDerivative) {

  //  ProcessThreaded(inputs);

  //  std::vector<double> last_dLoss_dOut;
  //  std::vector<double> next_dLoss_dOut;
  //  std::vector<double> last_dOut_dActivation;
  //  std::vector<double> next_dOut_dActivation;

  //  ThreadPool * pool = GetCpuSizedThreadPool();

  //  for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
  //    // Go backwards through the layers
  //    const std::size_t current = hiddenLayers_ - i;

  //    auto & layer = layers_[current];

  //    next_dLoss_dOut.resize(layer.size_);
  //    next_dOut_dActivation.resize(layer.size_);

  //    BatchTasks tasks(*pool);
  //    tasks.CreateBatches(layer.size_, [this, &layer, &next_dLoss_dOut,
  //      &next_dOut_dActivation, &lossFunctionDerivative, current]
  //      (std::size_t start, std::size_t end) {

  //      for (std::size_t k = start; k < end; ++k) {
  //        const std::size_t neuroneIndex = layer.size_ - 1 - k;
  //        auto & neurone = layer.neurones_[neuroneIndex];

  //        const double out = buffers_[current + 1][neuroneIndex];

  //        // dLoss/dWeight = dLoss/dOut * dOut/dActivation * dActivation/dWeight

  //        const double dLoss_dOut = lossFunctionDerivative(out, neuroneIndex);
  //        next_dLoss_dOut[k] = dLoss_dOut;

  //        // derivative of activation function
  //        const double dOut_dActivation =
  //          ActivationFunctionDerivative(layer.activationType_, out);

  //        next_dOut_dActivation[k] = dOut_dActivation;
  //      }
  //    });

  //    tasks.Run();

  //    last_dLoss_dOut = std::move(next_dLoss_dOut);
  //    std::reverse(last_dLoss_dOut.begin(), last_dLoss_dOut.end());
  //    next_dLoss_dOut.clear();

  //    last_dOut_dActivation = std::move(next_dOut_dActivation);
  //    std::reverse(last_dOut_dActivation.begin(), last_dOut_dActivation.end());
  //    next_dOut_dActivation.clear();

  //    if (current != 0) {
  //      lossFunctionDerivative =
  //        [this, &last_dLoss_dOut, &last_dOut_dActivation, current]
  //        (double, std::size_t index) {

  //          auto & layer = layers_[current];

  //          double value = 0.0;

  //          for (std::size_t i = 0; i < layer.size_; ++i) {
  //            value += last_dLoss_dOut[i] * last_dOut_dActivation[i]
  //              * layer.neurones_[i].weights_[index];
  //          }

  //          return value;
  //        };
  //    }
  //    else {
  //      lossFunctionDerivative =
  //        [this, last_dLoss_dOut, last_dOut_dActivation]
  //        (double, std::size_t index) {

  //          auto & layer = layers_[0];

  //          double value = 0.0;

  //          for (std::size_t i = 0; i < layer.size_; ++i) {
  //            value += last_dLoss_dOut[i] * last_dOut_dActivation[i]
  //              * layer.neurones_[i].weights_[index];
  //          }

  //          return value;
  //        };
  //    }
  //  }

  //  return lossFunctionDerivative;
  //}

private:
  void Create() {
    CHECK(hiddenLayers_ > 0);

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

  double learningRate_ = 0.5;
};

class HalvingLearningRate {
public:
  HalvingLearningRate(double learningRate)
    : learningRate_(learningRate) {}

  double LearningRate() const { return learningRate_; }
  void Advance() { learningRate_ /= 2.0; }

private:
  double learningRate_;
};

class SteppingLearningRate {
public:
  SteppingLearningRate(double learningRate, double step)
    : learningRate_(learningRate), step_(step) {}

  double LearningRate() const { return learningRate_; }
  void Advance() { learningRate_ -= step_; }

private:
  double learningRate_;
  double step_;
};

template<typename F, typename L>
void TrainNeuralNet(
  NeuralNet * net,
  F func,
  L learningRate,
  std::size_t epochs)
{
  for (std::size_t i = 0; i < epochs; ++i) {
    net->SetLearningRate(learningRate.LearningRate());
    func();
    learningRate.Advance();
  }
}
