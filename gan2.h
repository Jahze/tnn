#pragma once

#include <iterator>
#include <fstream>
#include "graphics.h"
#include "mnist.h"

#define GRAPH_LOSS 0

namespace mnist {

class GAN2 : public ::SimpleSimulation {
public:
  GAN2(std::size_t msPerFrame,
      OpenGLContext & context,
      const std::string & trainingFilename,
      const std::string & trainingLabels,
      const std::string & classifyFilename,
      const std::string & classifyLabels)
    : ::SimpleSimulation(msPerFrame)
    , context_(context), rng_(random_()) {

    scene_.reset(new Scene());

    trainingImages_ = mnist::ImageFile::Read(trainingFilename);

    CHECK(trainingImages_.Size() > 0);
    CHECK(trainingImages_.Width() == trainingImages_.Height());

    trainingOutput_ = mnist::LabelFile::Read(trainingLabels);

    CHECK(trainingOutput_.Size() == trainingImages_.Size());

    const std::size_t dimension = trainingImages_.Width();
    brain_.reset(new NeuralNet(dimension * dimension + 10, 1, 1, 300));
    //brain_->SetHiddenLayerActivationType(ActivationType::Tanh);
    brain_->SetHiddenLayerActivationType(ActivationType::LeakyReLu);
    brain_->SetLearningRate(0.002);
    generator_.reset(new NeuralNet(NoiseInputs + 10, dimension * dimension, 1, 100));
    generator_->SetHiddenLayerActivationType(ActivationType::LeakyReLu);
    //generator_->SetHiddenLayerActivationType(ActivationType::Tanh);
    generator_->SetLearningRate(0.001);

    classifyImages_ = mnist::ImageFile::Read(classifyFilename);

    CHECK(classifyImages_.Size() > 0);
    CHECK(classifyImages_.Width() == classifyImages_.Height());
    CHECK(classifyImages_.Width() == trainingImages_.Width());

    classifyOutput_ = mnist::LabelFile::Read(classifyLabels);

    CHECK(classifyOutput_.Size() == classifyImages_.Size());

    RenderImages();

    context.AddResizeListener([this](){
      recreateScene_ = true; });

    brain_->DeserializeWeights("discriminator-weights.txt");
    generator_->DeserializeWeights("generator-weights.txt");

    context.AddDestroyListener([this](){
      brain_->SerializeWeights("discriminator-weights.txt");
      generator_->SerializeWeights("generator-weights.txt");
    });

#if GRAPH_LOSS
    Graph::Limits limits;
    limits.xmin = 0.0; limits.xmax = 1000.0;
    limits.ymin = 0.0; limits.ymax = 10.0;

    graphWindow_ = std::make_unique<GraphWindow>(limits);

    fakeLossSeries_.r = 0;
    fakeLossSeries_.b = 255;

    context.AddResizeListener([this](){
      graphWindow_->Graph()->SignalRedraw(); });
#endif
  }

protected:
  void StartImpl() {
  }

  void UpdateImpl(bool render, std::size_t ms) {
    if (render && recreateScene_) {
      DestroyScene();
      RenderImages();
      recreateScene_ = false;
    }

    if (trainingCursor_ + BatchSize >= trainingImages_.NormalisedData().size())
      trainingCursor_ = 0u;

    Timer timer;

    std::vector<std::vector<double>> idealOutputs(BatchSize);
    for (auto && element : idealOutputs)
      element.reserve(1u);

    const auto & loss = TrainDiscriminator(idealOutputs);
    TrainGenerator(idealOutputs);

    std::cout << "Iteration " << ++iteration_
      << " [" << timer.ElapsedMicroseconds() << "]\n";

#if GRAPH_LOSS
    realLossSeries_.points.push_back({ static_cast<double>(iteration_), loss.first });
    fakeLossSeries_.points.push_back({ static_cast<double>(iteration_), loss.second });
    graphWindow_->Graph()->Clear();
    graphWindow_->Graph()->AddSeries(realLossSeries_);
    graphWindow_->Graph()->AddSeries(fakeLossSeries_);
#endif
    recreateScene_ = true;

    scene_->Update(ms);

    if (render) {
      context_.MakeActive();
      scene_->Render(context_);

#if GRAPH_LOSS
      graphWindow_->Context()->MakeActive();
      ::PAINTSTRUCT ps;
      ::HDC hdc = ::BeginPaint(graphWindow_->Context()->Handle(), &ps);
      graphWindow_->Graph()->DrawAxes();
      graphWindow_->Graph()->DrawSeries();
      ::EndPaint(context_.Handle(), &ps);
#endif
    }
  }

  std::vector<double> GeneratorInputs(const std::vector<double> & outputs) {
    std::vector<double> generatorInputs;
    std::generate_n(std::back_inserter(generatorInputs), NoiseInputs,
      [this]() {return RandomDouble(-1.0, 1.0); });

    // Put one-hot vector as input to switch type of digit generation
    for (std::size_t i = 0u; i < 10u; ++i)
      generatorInputs.push_back(outputs[i]);

    return generatorInputs;
  }

  void TrainGenerator(std::vector<std::vector<double>> & idealOutputs) {
    const auto & labels = trainingOutput_.Data();

    std::vector<std::vector<double>> inputs(BatchSize);

    for (std::size_t i = 0u; i < BatchSize; ++i) {
      const std::size_t trainingCursor = trainingCursor_ + i;

      const auto & label = labels[trainingCursor];

      auto generatorInputs = GeneratorInputs(label);

      inputs[i] = std::move(generator_->ProcessThreaded(generatorInputs));
      inputs[i].insert(inputs[i].end(), label.begin(), label.end());

      idealOutputs[i][0] = 1.0;
    }

    generator_->BackPropagationCrossEntropy(*brain_.get(),
      inputs, idealOutputs);

    trainingCursor_ += 100u;
  }

  std::pair<double,double> TrainDiscriminator(
      std::vector<std::vector<double>> & idealOutputs) {

    double realLoss = 0.0;
    double fakeLoss = 0.0;

    const auto & normalisedImages = trainingImages_.NormalisedData();
    const auto & labels = trainingOutput_.Data();

    const std::size_t dimension = trainingImages_.Width();
    const std::size_t inputSize = dimension * dimension;

    std::vector<std::vector<double>> inputs(BatchSize);
    for (std::size_t i = 0u; i < BatchSize; ++i)
      inputs[i].resize(inputSize);

    for (std::size_t i = 0u; i < BatchSize; ++i) {
      const std::size_t trainingCursor = trainingCursor_ + i;

      const auto & label = labels[trainingCursor];

      const auto & normalisedImage = normalisedImages[trainingCursor];
      inputs[i].assign(
        normalisedImage.inputs.get(),
        normalisedImage.inputs.get() + inputSize);

      inputs[i].insert(inputs[i].end(), label.begin(), label.end());

      // Want a value of 1.0 (which says it is a digit
      // with a probability of 100%)
      idealOutputs[i][0] = 1.0;
    }

    brain_->BackPropagationCrossEntropy(inputs, idealOutputs);

    for (std::size_t i = 0u; i < BatchSize; ++i) {
      const std::size_t trainingCursor = trainingCursor_ + i;

      const auto & label = labels[trainingCursor];

      std::vector<double> generatorInputs = GeneratorInputs(label);

      inputs[i] = std::move(generator_->ProcessThreaded(generatorInputs));
      inputs[i].insert(inputs[i].end(), label.begin(), label.end());

      // Want a value of 0.0 (because it's not from the training set)
      idealOutputs[i][0] = 0.0;
    }

    brain_->BackPropagationCrossEntropy(inputs, idealOutputs);

    trainingCursor_ += 100u;
#if GRAPH_LOSS
      const auto & realOutputs = brain_->ProcessThreaded(inputs);
      const auto & fakeOutputs = brain_->ProcessThreaded(generatedOutputs);
      const double realOutputLoss = 1.0 - realOutputs[0];
      const double fakeOutputLoss = 0.0 - fakeOutputs[0];

      realLoss += std::abs(realOutputLoss);
      fakeLoss += std::abs(fakeOutputLoss);
#endif

    return {realLoss, fakeLoss};
  }

  void DestroyScene() {
    classifyIndex_ = 0;
    scene_.reset(new Scene());
    objects_.clear();
    classifyData_.clear();
  }

  void RenderImages() {
    const auto & images = classifyImages_.Data();
    const auto & normalizedImages = classifyImages_.NormalisedData();
    const std::size_t size = images.size();

    const std::size_t dimension = classifyImages_.Width();
    const std::size_t normalizedSize = dimension * dimension;
    const std::size_t pixels = 3 * dimension * dimension;

    const std::size_t perRow = context_.Width() / dimension;
    const std::size_t perColumn = context_.Height() / dimension;
    const double cellDistanceX = 2.0 / (double)perRow;
    const double cellDistanceY = 2.0 / (double)perColumn;
    std::size_t row = 0;
    std::size_t column = 0;

    bool useGenerated = false;

    const auto & labels = classifyOutput_.Data();
    for (std::size_t i = 0; i < size && row < perRow; ++i) {
      double x = -1.0 + (cellDistanceX * column++);
      double y = -1.0 + (cellDistanceY * row);

      if (column == perRow) {
        column = 0;
        row++;
      }

      std::unique_ptr<double[]> normalizedData(new double[normalizedSize + 10]);
      std::unique_ptr<uint8_t[]> data(new uint8_t[pixels]);
      std::size_t dataIndex = 0;

      if (useGenerated) {
        const auto & label = labels[i];
        std::vector<double> generatorInputs = GeneratorInputs(labels[i]);

        auto outputs = generator_->ProcessThreaded(generatorInputs);

        for (std::size_t j = 0, length = outputs.size(); j < length; ++j) {
          data[dataIndex++] = (uint8_t)(outputs[j] * 255.0);
          data[dataIndex++] = (uint8_t)(outputs[j] * 255.0);
          data[dataIndex++] = (uint8_t)(outputs[j] * 255.0);
          // TODO: image orientation doesn't match outputs? [see other case]
          normalizedData[j] = outputs[j];
        }

        std::copy(label.begin(), label.end(),
          normalizedData.get() + outputs.size());
      }
      else {
        const auto & image = images[i];

        for (std::size_t j = 0; j < dimension; ++j) {
          for (std::size_t k = 0; k < dimension; ++k) {
            const std::size_t byte = dimension - j - 1;
            data[dataIndex++] = image.bytes[k + byte * dimension];
            data[dataIndex++] = image.bytes[k + byte * dimension];
            data[dataIndex++] = image.bytes[k + byte * dimension];
            normalizedData[dataIndex / 3] =
              normalizedImages[i].inputs[dataIndex / 3];
          }
        }

        const auto & label = labels[i];
        std::copy(label.begin(), label.end(),
          normalizedData.get() + normalizedSize);
      }

      CHECK(dataIndex == pixels);

      objects_.emplace_back(new Glyph(x, y, data.get(), dimension));
      auto & object = objects_.back();
      scene_->AddObject(object.get());

      classifyData_.emplace_back(normalizedData.release());
      useGenerated = !useGenerated;
    }

    for (std::size_t i = 0u; i < objects_.size(); ++i) {
      const std::size_t dimension = classifyImages_.Width();
      const std::size_t inputSize = dimension * dimension + 10;

      std::vector<double> inputs(
        classifyData_[i].get(),
        classifyData_[i].get() + inputSize);

      auto outputs = brain_->ProcessThreaded(inputs);
      objects_[i]->Confidence(outputs[0]);
    }
  }

private:
  double RandomDouble(double min, double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(rng_);
  }

private:
  std::random_device random_;
  std::mt19937 rng_;

  OpenGLContext & context_;
  std::unique_ptr<Scene> scene_;
  std::vector<std::unique_ptr<Glyph>> objects_;
  std::unique_ptr<NeuralNet> brain_;
  std::unique_ptr<NeuralNet> generator_;
  ImageFile trainingImages_;
  LabelFile trainingOutput_;
  ImageFile classifyImages_;
  LabelFile classifyOutput_;
  std::vector<std::unique_ptr<double[]>> classifyData_;
  std::size_t classifyIndex_ = 0u;
  bool recreateScene_ = false;

  std::unique_ptr<GraphWindow> graphWindow_;
  Graph::Series realLossSeries_;
  Graph::Series fakeLossSeries_;

  std::size_t iteration_ = 0u;
  std::size_t trainingCursor_ = 0u;

  const std::size_t NoiseInputs = 100u;
  const std::size_t BatchSize = 100u;
};

}
