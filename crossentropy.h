#pragma once

#include <iterator>
#include "graph.h"
#include "graphics.h"
#include "mnist.h"

class CrossEntropy : public ::SimpleSimulation {
public:
  CrossEntropy(std::size_t msPerFrame,
      OpenGLContext & context)
    : ::SimpleSimulation(msPerFrame)
    , context_(context), xmin_(-1.0), xmax_(1.0) {

    brain_.reset(new NeuralNet(1, 1, 1, 2));
    brain_->SetLearningRate(0.01);
    brain_->SetHiddenLayerActivationType(ActivationType::Tanh);

    //std::vector<double> weights = {
    //  2.29015, 0, -0.927542, 0, -2.04721, -1.12544, 0
    //};

    //brain_->SetWeights(weights);

    Graph::Limits limits{xmin_, xmax_, 0.0, 1.0};
    graph_.reset(new Graph(context_.Handle(), limits));

    GenerateData();
    AddClassifyData();

    RecordWeights();

    context_.AddResizeListener(std::bind(&CrossEntropy::Resize, this));
  }

protected:
  void Resize() {
    graph_->SignalRedraw();
  }

  void StartImpl() {
  }

  void UpdateImpl(bool render, std::size_t ms) {
    graph_->Clear();

    AddClassifyData();

    Train(10u);

    if (render) Draw();
  }

  void Train(std::size_t epochs) {
    const std::size_t cases = 100u;
    const double min = xmin_;
    const double max = xmax_;
    double increment = (max - min) / cases;
    std::vector<double> inputs(1);
    Aligned32ByteRAIIStorage<double> idealOutputs(1);

    std::vector<double> trainingInputs;
    for (double input = min; input < max; input += increment)
      trainingInputs.push_back(input);

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
      //std::random_shuffle(trainingInputs.begin(), trainingInputs.end());
      for (auto && input : trainingInputs) {
        double normalised = (input - min) / (max - min);
        double normalisedValue = -1.0 + normalised * 2.0;
        inputs[0] = normalisedValue;
        idealOutputs[0] = TargetFunction(input);
        brain_->BackPropagationCrossEntropy(inputs, idealOutputs);
      }
      epochs_++;
    }

    Graph::Series series;
    series.r = 0;
    series.b = 255;

    loss_ = 0.0;
    increment = (max - min) / 100u;
    //for (double input = min; input < max; input += increment) {
    for (auto && input : trainingInputs) {
      double normalised = (input - min) / (max - min);
      double normalisedValue = -1.0 + normalised * 2.0;
      inputs[0] = normalisedValue;

      auto outputs = brain_->ProcessThreaded(inputs);
      series.points.push_back({ normalisedValue, outputs[0] });

      double loss = outputs[0] - TargetFunction(input);
      loss_ += loss * loss;
    }

    graph_->AddSeries(series);
  }

  void Draw() {
    ::PAINTSTRUCT ps;
    ::HDC hdc = ::BeginPaint(context_.Handle(), &ps);

    ::FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));

    graph_->DrawAxes();
    graph_->DrawSeries();

    auto oldColour = ::GetBkColor(hdc);
    ::SetBkColor(hdc, RGB(128,128,128));

    std::string epoch = "Epoch: " + std::to_string(epochs_)
      + " Loss: " + std::to_string(loss_);

    ::TextOut(hdc, 20, 0, epoch.c_str(), (int)epoch.size());

    ::SetBkColor(hdc, oldColour);

    ::EndPaint(context_.Handle(), &ps);
  }

  void GenerateData() {
    const std::size_t cases = 100u;
    const double min = xmin_;
    const double max = xmax_;
    double increment = (max - min) / cases;

    for (double input = min; input < max; input += increment) {
      LabelledData data = { input, TargetFunction(input) };
      m_data.emplace_back(data);
    }
  }

  void AddClassifyData() {
    Graph::Series series;
    for (auto && datum : m_data) {
      series.points.push_back({ datum.input, datum.output });
    }
    graph_->AddSeries(series);
  }

  double TargetFunction(double input) {
    return 1.0 / (1.0 + std::exp(-10.0 * input));
    //return input > 0.0 ? 1.0 : 0.0;
    //return input;
    //return 0.5;
  }

private:
  double RandomDouble(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

  void PrintWeights(std::ostream & stream) {
    bool first = true;
    for (auto && weight : brain_->GetWeights()) {
      if (!first) stream << ", ";
      first = false;
      stream << weight;
    }
    stream << "\n";
  }

  void RecordWeights() {
    std::ofstream file("weights.txt");
    file << "INITIAL WEIGHTS: ";
    PrintWeights(file);

    const std::size_t cases = 100u;
    const double min = xmin_;
    const double max = xmax_;
    double increment = (max - min) / cases;
    std::vector<double> inputs(1);
    Aligned32ByteRAIIStorage<double> idealOutputs(1);

    std::vector<double> trainingInputs;
    for (double input = min; input < max; input += increment)
      trainingInputs.push_back(input);

    for (auto && input : trainingInputs) {
      double normalised = (input - min) / (max - min);
      double normalisedValue = -1.0 + normalised * 2.0;
      inputs[0] = normalisedValue;
      idealOutputs[0] = TargetFunction(input);
      brain_->BackPropagationCrossEntropy(inputs, idealOutputs);
      file << "input: " << input << " [" << normalisedValue << "]: ";
      PrintWeights(file);
    }
  }

private:
  std::random_device random_;
  std::mt19937 rng_;

  OpenGLContext & context_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<NeuralNet> brain_;
  std::size_t epochs_ = 0u;
  double loss_;
  double xmin_;
  double xmax_;

  struct LabelledData {
    double input;
    double output;
  };

  std::vector<LabelledData> m_data;
};
