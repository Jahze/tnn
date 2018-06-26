#pragma once

#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <vector>
#include "macros.h"
#include "matrix.h"
#include "threadpool.h"

enum class ActivationType {
  Sigmoid,
  Tanh,
  ReLu,
  Identity,
  LeakyReLu,
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
  AlignedMatrix weights_;
  AlignedMatrix transpose_;
  AlignedMatrix weightsDelta_;
  AlignedMatrix outputs_;
  AlignedMatrix dLoss_dNet_;
  AlignedMatrix * inputs_;
  ActivationType activationType_ = ActivationType::Sigmoid;

  NeuroneLayer(std::size_t size, std::size_t neuroneSize)
    : size_(size), neuroneSize_(neuroneSize)
    , weightsPerNeurone_(neuroneSize_ + 1) {
    // One extra for bias

    inputs_ = nullptr;
    weights_.Reset(size, weightsPerNeurone_);
    transpose_.Reset(size, weightsPerNeurone_);
    weightsDelta_.Reset(size, weightsPerNeurone_);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<> distribution(
      0.0, sqrt(2.0 / static_cast<double>(size)));

    for (std::size_t i = 0; i < size_; ++i) {
      const std::size_t base = i * weightsPerNeurone_;

      for (std::size_t j = 0; j < neuroneSize_; ++j)
        weights_.Value(i, j) = distribution(generator);

      // Initialise bias to 0
      weights_.Value(i, neuroneSize) = 0.0;
    }
  }

  void CreateOutputMatrix(std::size_t inputCount) {
    outputs_.Reset(inputCount, size_ + 1);
  }

  Neurone Neurones(std::size_t i) {
    return weights_.Row(i);
  }

  double Weights(std::size_t i, std::size_t j) const {
    return weights_.Value(i, j);
  }

  const static double kThresholdBias;

  void Process(AlignedMatrix & inputs) {
    weights_.Multiply(inputs, outputs_);

    for (std::size_t i = 0u; i < outputs_.rows_; ++i) {
      for (std::size_t j = 0u; j < size_; ++j)
        outputs_[i][j] = ActivationFunction(activationType_, outputs_[i][j]);

      outputs_[i][size_] = kThresholdBias;
    }

    inputs_ = &inputs;
  }

  void ProcessThreaded(AlignedMatrix & inputs) {
    if (size_ < 16u)
      weights_.Multiply(inputs, outputs_);
    else
      weights_.MultiplyThreaded(inputs, outputs_);

      for (std::size_t i = 0u; i < outputs_.rows_; ++i) {
        for (std::size_t j = 0u; j < size_; ++j)
          outputs_[i][j] = ActivationFunction(activationType_, outputs_[i][j]);

        outputs_[i][size_] = kThresholdBias;
      }

    inputs_ = &inputs;
  }

  void CommitDeltas() {
    if (size_ < 16u)
      weights_.Subtract(weightsDelta_);
    else
      weights_.SubtractThreaded(weightsDelta_);

    weightsDelta_.Zero();
  }
};

class NeuralNet {
public:
  const static double kThresholdBias;

  enum class UpdateType { Stochastic, Batched };

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

  AlignedMatrix Process(const std::vector<std::vector<double>> & batch) {
    const std::size_t inputSize = batch[0].size();

    inputsStorage_.Reset(batch.size(), inputSize + 1);

    for (std::size_t i = 0, length = batch.size(); i < length; ++i) {
      std::copy(batch[i].cbegin(), batch[i].cend(), inputsStorage_[i]);
      inputsStorage_[i][inputSize] = kThresholdBias;
    }

    AlignedMatrix * in = &inputsStorage_;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      layers_[i].CreateOutputMatrix(batch.size());
      layers_[i].Process(*in);

      in = &layers_[i].outputs_;
    }

    return std::move(in->Clone());
  }

  std::vector<double> Process(const std::vector<double> & inputs) {
    CHECK(inputs.size() == inputs_);

    std::vector<std::vector<double>> batch = {inputs};

    const AlignedMatrix & outputs = Process(batch);

    return { outputs[0], outputs[0] + layers_.back().size_ };
  }

  AlignedMatrix ProcessThreaded(
      const std::vector<std::vector<double>> & batch) {

    const std::size_t inputSize = batch[0].size();

    inputsStorage_.Reset(batch.size(), inputSize + 1);

    for (std::size_t i = 0, length = batch.size(); i < length; ++i) {
      std::copy(batch[i].cbegin(), batch[i].cend(), inputsStorage_[i]);
      inputsStorage_[i][inputSize] = kThresholdBias;
    }

    AlignedMatrix * in = &inputsStorage_;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      layers_[i].CreateOutputMatrix(batch.size());
      layers_[i].ProcessThreaded(*in);

      in = &layers_[i].outputs_;
    }

    return std::move(in->Clone());
  }

  std::vector<double> ProcessThreaded(const std::vector<double> & inputs) {
    CHECK(inputs.size() == inputs_);

    std::vector<std::vector<double>> batch = {inputs};

    const auto & outputs = ProcessThreaded(batch);

    return { outputs[0], outputs[0] + layers_.back().size_ };
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

  void SerializeWeights() {
    std::ofstream file("weights.txt");
    for (auto && weight : GetWeights())
      file << weight << "\n";
  }

  void DeserializeWeights() {
    std::vector<double> weights;
    std::ifstream file("weights.txt");
    while (!file.eof()) {
      std::string line;
      std::getline(file, line);
      if (line.empty()) continue;
      weights.push_back(std::stod(line));
    }
    SetWeights(weights);
  }

  void SetLearningRate(double rate) {
    learningRate_ = rate;
  }

  void SetHiddenLayerActivationType(ActivationType type) {
    // Tanh needs a lower learning rate generally
    for (std::size_t i = 0; i < hiddenLayers_; ++i)
      layers_[i].activationType_ = type;
  }

  void SetOutputLayerActivationType(ActivationType type) {
    layers_[hiddenLayers_].activationType_ = type;
  }

  void SetUpdateType(UpdateType type) {
    updateType_ = type;
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
        const double out = layers_[current].outputs_[0][neuroneIndex];

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
            * (*layers_[current].inputs_)[0][weightIndex];

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

  void BackPropagationThreaded(const std::vector<std::vector<double>> & inputs,
    const std::vector<std::vector<double>> & idealOutputs) {

    const auto & outputs = ProcessThreaded(inputs);

    const std::size_t length = layers_.back().size_;
    const std::size_t batchSize = inputs.size();

    AlignedMatrix loss{batchSize, length};

    for (std::size_t i = 0u; i < batchSize; ++i) {
      for (std::size_t j = 0u; j < length; ++j)
        loss[i][j] = -(idealOutputs[i][j] - outputs[i][j]);
    }

    Calculate_dLoss_dNet(hiddenLayers_, loss);

    CalculateGradients();
    UpdateWeights();
  }

  void BackPropagationCrossEntropy(const std::vector<double> & inputs,
      Aligned32ByteRAIIStorage<double> & idealOutputs) {

    const auto & outputs = ProcessThreaded(inputs);

    const std::size_t length = outputs.size();
    const std::size_t inputCount = 1u;

    layers_[hiddenLayers_].dLoss_dNet_.Reset(inputCount, length);

    for (std::size_t i = 0u; i < inputCount; ++i) {
      double * dLoss_dNet = layers_[hiddenLayers_].dLoss_dNet_[i];

      // TODO: needed?
      //std::memset(dLoss_dNet, 0,
      //  sizeof(double) * layers_[hiddenLayers_].dLoss_dNet_[i].alignedColumns_);

      // cross-entropy loss
      // https://www.ics.uci.edu/~pjsadows/notes.pdf
      for (std::size_t i = 0u; i < length; ++i)
        dLoss_dNet[i] = (outputs[i] - idealOutputs[i]);
    }

    CalculateGradients();
    UpdateWeights();
  }

  void BackPropagationCrossEntropy(NeuralNet & net,
      const std::vector<double> & inputs,
      Aligned32ByteRAIIStorage<double> & idealOutputs) {

    return;
    //const auto & outputs = net.ProcessThreaded(inputs);

    //const std::size_t length = outputs.size();

    //net.layers_[hiddenLayers_].dLoss_dNet_.Reset(length);
    //double * dLoss_dNet = net.layers_[hiddenLayers_].dLoss_dNet_.Get();

    //// cross-entropy loss
    //// https://www.ics.uci.edu/~pjsadows/notes.pdf
    //for (std::size_t i = 0u; i < length; ++i)
    //  dLoss_dNet[i] = (outputs[i] - idealOutputs[i]);

    //// Calculate dLoss_dNet for discriminator
    //net.CalculateGradients();

    //// Get final loss to calculate genertor dLoss_dNet
    //AlignedMatrix lastLoss;
    //net.CalculateLoss(0, lastLoss);

    //Calculate_dLoss_dNet(hiddenLayers_, lastLoss);

    //// TODO: segfault here sometimes
    //CalculateGradients();
    //UpdateWeights();
  }

  void CommitDeltas() {
    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i)
      layers_[i].CommitDeltas();
  }

  // TODO: debugging temp
  const std::vector<NeuroneLayer> & Layers() const { return layers_; }

private:
  void Create() {
    CHECK(hiddenLayers_ > 0);

    layers_.emplace_back(neuronesPerHiddenLayer_, inputs_);

    for (std::size_t i = 0, length = hiddenLayers_ - 1; i < length; ++i)
      layers_.emplace_back(neuronesPerHiddenLayer_, neuronesPerHiddenLayer_);

    layers_.emplace_back(outputs_, neuronesPerHiddenLayer_);
  }

  void UpdateWeights() {
    Aligned32ByteRAIIStorage<double> lastLoss;

    ThreadPool * pool = GetCpuSizedThreadPool();

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      // Go backwards through the layers
      const std::size_t current = hiddenLayers_ - i;

      auto & layer = layers_[current];

      using namespace std::placeholders;

      BatchTasks tasks(*pool);
      tasks.CreateBatches(layer.size_, updateType_ == UpdateType::Stochastic
        ? std::bind(&NeuralNet::UpdateWeightsStochastic,
            this, std::ref(layer), _1, _2)
        : std::bind(&NeuralNet::UpdateWeightsBatched,
            this, std::ref(layer), _1, _2));

      tasks.Run();
    }
  }

  void UpdateWeightsStochastic(NeuroneLayer & layer,
      std::size_t start, std::size_t end) {

    const double LearningRate = learningRate_;

    // dLoss/dWeight = dLoss/dOut * dOut/dNet * dNet/dWeight

    const double * dLoss_dNet = layer.dLoss_dNet_[0];
    const double * dNet_dWeights = (*layer.inputs_)[0];

    for (std::size_t k = start; k < end; ++k) {
      auto & neurone = layer.Neurones(k);
      const std::size_t inputs = layer.weightsPerNeurone_;

      __m256d learningRate = _mm256_set1_pd(LearningRate * dLoss_dNet[k]);

      const std::size_t batches = AlignTo32Bytes<double>(inputs) / 4;
      for (std::size_t l = 0; l < batches; ++l) {
        __m256d dNet_dWeight = _mm256_load_pd(dNet_dWeights + (l * 4));
        __m256d product = _mm256_mul_pd(dNet_dWeight, learningRate);
        __m256d neuroneWeights = _mm256_load_pd(neurone.ptr_ + (l * 4));
        __m256d result = _mm256_sub_pd(neuroneWeights, product);
        std::memcpy(neurone.ptr_ + (l * 4),
          result.m256d_f64, sizeof(double) * 4);
      }
    }
  }

  void UpdateWeightsBatched(NeuroneLayer & layer,
      std::size_t start, std::size_t end) {

    const double LearningRate = learningRate_;

    // dLoss/dWeight = dLoss/dOut * dOut/dNet * dNet/dWeight

    const std::size_t batchSize = layer.outputs_.rows_;
    const std::size_t inputs = layer.weightsPerNeurone_;

    for (std::size_t i = 0u; i < batchSize; ++i) {
      const double * dLoss_dNet = layer.dLoss_dNet_[i];
      const double * dNet_dWeights = (*layer.inputs_)[i];

      for (std::size_t k = start; k < end; ++k) {
        auto & neurone = layer.Neurones(k);

        double * weightDelta = layer.weightsDelta_.Row(k);

        //__m256d learningRate = _mm256_set1_pd(LearningRate * dLoss_dNet[k]);

        //const std::size_t batches = AlignTo32Bytes<double>(inputs) / 4;
        //for (std::size_t l = 0; l < batches; ++l) {
        //  __m256d dNet_dWeight = _mm256_load_pd(dNet_dWeights + (l * 4));
        //  __m256d product = _mm256_mul_pd(dNet_dWeight, learningRate);
        //  __m256d lastWeights = _mm256_load_pd(weightDelta + (l * 4));
        //  __m256d nextWeights = _mm256_add_pd(lastWeights, product);
        //  std::memcpy(weightDelta + (l * 4),
        //    nextWeights.m256d_f64, sizeof(double) * 4);
        //}

        // NB: vector instructions make this loop slightly slower
        const double learningRate = dLoss_dNet[k] * LearningRate;

        for (std::size_t j = 0; j < inputs; ++j) {
          double delta = dNet_dWeights[j] * learningRate;
          weightDelta[j] += delta;
        }
      }
    }
  }

  void CalculateGradients() {
    AlignedMatrix lastLoss;

    for (std::size_t i = 0, length = hiddenLayers_ + 1; i < length; ++i) {
      // Go backwards through the layers
      const std::size_t current = hiddenLayers_ - i;

      auto & layer = layers_[current];

      if (current > 0u) {
        CalculateLoss(current, lastLoss);
        Calculate_dLoss_dNet(current - 1, lastLoss);
      }
    }
  }

  void Calculate_dLoss_dNet(std::size_t current, AlignedMatrix & loss) {
    auto & layer = layers_[current];

    const std::size_t batchSize = layer.outputs_.rows_;

    layer.dLoss_dNet_.Reset(batchSize, layer.size_);

    for (std::size_t i = 0u; i < batchSize; ++i) {
      for (std::size_t k = 0u; k < layer.size_; ++k) {
        const double out = layer.outputs_[i][k];

        layer.dLoss_dNet_[i][k] =
          ActivationFunctionDerivative(layer.activationType_, out);
      }

      SIMDMultiply(loss[i], layer.dLoss_dNet_[i],
        layer.dLoss_dNet_[i], layer.size_);
    }

  }

  void CalculateLoss(std::size_t current,
    AlignedMatrix & loss) {

    auto & layer = layers_[current];

    const std::size_t layerSize = layer.size_;
    const std::size_t prevLayerSize = layer.neuroneSize_;
    const std::size_t batchSize = layer.outputs_.rows_;

    loss.Reset(batchSize, prevLayerSize);

    for (std::size_t input = 0u; input < batchSize; ++input) {
      double * dLoss_dNet = layer.dLoss_dNet_[input];

      for (std::size_t i = 0u; i < prevLayerSize; ++i) {
        for (std::size_t j = 0u; j < layerSize; ++j) {
          loss[input][i] += dLoss_dNet[j] * layer.Weights(j, i);
        }
      }
    }
  }

private:
  std::size_t inputs_;
  std::size_t outputs_;
  std::size_t hiddenLayers_;
  std::size_t neuronesPerHiddenLayer_;

  std::vector<NeuroneLayer> layers_;

  AlignedMatrix inputsStorage_;

  UpdateType updateType_ = UpdateType::Stochastic;
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
