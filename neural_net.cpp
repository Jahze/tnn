#include "neural_net.h"

const double NeuroneLayer::kThresholdBias = -1.0;
const double NeuralNet::kThresholdBias = -1.0;

namespace {
  const double kActivationResponse = 1.0;
}

double Sigmoid(double activation, double response) {
  return (1.0 / (1.0 + std::exp(-activation / response)));
}

double Tanh(double activation) {
  return tanh(activation);
}

void ActivationFunction(
    ActivationType type,
    double * outputs,
    std::size_t size) {

  double totalE = 0.0;
  std::unique_ptr<double[]> outputsE;

  if (type == ActivationType::Softmax) {
    // std::exp will reach inf when x > ~700. We can ensure numerical stability
    // by subtracting the maximum output value from each x before we call
    // std::exp. This is mathematically equivalent to vanilla softmax.

    double max = -std::numeric_limits<double>::max();

    for (std::size_t i = 0u; i < size; ++i) {
      if (outputs[i] > max)
        max = outputs[i];
    }

    outputsE = std::make_unique<double[]>(size);

    for (std::size_t i = 0u; i < size; ++i) {
      outputsE[i] = std::exp(outputs[i] - max);
      totalE += outputsE[i];
    }
  }

  for (std::size_t i = 0u; i < size; ++i) {
    const double activation = outputs[i];

    switch (type) {
    case ActivationType::Sigmoid:
      outputs[i] = Sigmoid(activation, kActivationResponse);
      continue;
    case ActivationType::Tanh:
      outputs[i] = Tanh(activation);
      continue;
    case ActivationType::ReLu:
      outputs[i] = activation >= 0.0 ? activation : 0.0;
      continue;
    case ActivationType::Identity:
      outputs[i] = activation;
      continue;
    case ActivationType::LeakyReLu:
      outputs[i] =  activation >= 0.0 ? activation : activation*0.01;
      continue;
    case ActivationType::Softmax:
      outputs[i] = outputsE[i] / totalE;
      continue;
    }
  }
}

double ActivationFunctionDerivative(ActivationType type, double value) {
  switch (type) {
  case ActivationType::Sigmoid:
    return value * (1.0 - value);
  case ActivationType::Tanh:
    return 1.0 - (value * value);
  case ActivationType::ReLu:
    return value >= 0.0 ? 1.0 : 0.0;
  case ActivationType::Identity:
    return 1.0;
  case ActivationType::LeakyReLu:
    return value >= 0.0 ? 1.0 : 0.01;
  }
}
