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
  if (type == ActivationType::Softmax) {
    for (std::size_t i = 0u; i < size; ++i)
      totalE += std::exp(outputs[i]);
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
      outputs[i] = std::exp(activation) / totalE;
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
