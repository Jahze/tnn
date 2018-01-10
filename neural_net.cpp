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

double ActivationFunction(ActivationType type, double activation) {
  switch (type) {
  case ActivationType::Sigmoid:
    return Sigmoid(activation, kActivationResponse);
  case ActivationType::Tanh:
    return Tanh(activation);
  case ActivationType::ReLu:
    return activation >= 0.0 ? activation : 0.0;
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
  }
}
