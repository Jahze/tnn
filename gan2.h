#pragma once

#include <iterator>
#include "graphics.h"
#include "mnist.h"

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
    brain_.reset(new NeuralNet(dimension * dimension, 1, 1, 300));
    brain_->SetHiddenLayerActivationType(ActivationType::Tanh);
    generator_.reset(new NeuralNet(100 + 10, dimension * dimension, 1, 100));
    //generator_->SetHiddenLayerActivationType(ActivationType::ReLu);
    generator_->SetHiddenLayerActivationType(ActivationType::Tanh);
    generator_->SetLearningRate(0.01);

    TrainModel();

    classifyImages_ = mnist::ImageFile::Read(classifyFilename);

    CHECK(classifyImages_.Size() > 0);
    CHECK(classifyImages_.Width() == classifyImages_.Height());
    CHECK(classifyImages_.Width() == trainingImages_.Width());

    classifyOutput_ = mnist::LabelFile::Read(classifyLabels);

    CHECK(classifyOutput_.Size() == classifyImages_.Size());

    context.AddResizeListener([this](){ recreateScene_ = true; });
    RenderImages();
  }

protected:
  void StartImpl() {
    generation_ = 0;
  }

  void UpdateImpl(bool render, std::size_t ms) {
    if (recreateScene_) {
      DestroyScene();
      RenderImages();
      recreateScene_ = false;
    }

    //for (std::size_t i = 0u; i < 50u; ++i) {
    //  TrainOne(trainingCursor_++);
    //  if (trainingCursor_ % 1000u == 0)
    //    recreateScene_ = true;
    //}

    //recreateScene_ = true;

    scene_->Update(ms);

    if (render)
      scene_->Render(context_);
  }

  std::vector<double> GeneratorInputs(const std::vector<double> & outputs) {
    std::vector<double> generatorInputs;
    std::generate_n(std::back_inserter(generatorInputs), 100,
      //[this]() {return RandomDouble(0.0, 1.0); });
      [this]() {return RandomDouble(-1.0, 1.0); });

    // Put one-hot vector as input to switch type of digit generation
    for (std::size_t i = 0u; i < 10u; ++i)
      generatorInputs.push_back(outputs[i]);

    return generatorInputs;
  }

  void TrainModel() {
    auto TrainFunction = [this]() {
      const auto & normalisedImages = trainingImages_.NormalisedData();
      const std::size_t dimension = trainingImages_.Width();
      const std::size_t inputSize = dimension * dimension;

      std::vector<double> inputs(inputSize);

      std::cout << "Training.....0%";

      const std::size_t length = normalisedImages.size();
      const std::size_t one_hundredth = length / 100u;
      std::size_t progress = 0u;
      std::size_t percent = 0u;

      Timer trainingTimer;

      Aligned32ByteRAIIStorage<double> idealOutputs(1u);
      Aligned32ByteRAIIStorage<double> loss;

      const auto & data = trainingOutput_.Data();

      //for (std::size_t i = 0u; i < length; ++i, ++progress) {
      for (std::size_t i = 0u; i < 20000u; ++i, ++progress) {
        const auto & normalisedImage = normalisedImages[i];

        inputs.assign(
          normalisedImage.inputs.get(),
          normalisedImage.inputs.get() + inputSize);

        // Want a value of 1.0 (which says it is a digit
        // with a probability of 100%)
        idealOutputs[0] = 1.0;

        brain_->BackPropagationCrossEntropy(inputs, idealOutputs);

        const auto & outputs = data[i];
        std::vector<double> generatorInputs = GeneratorInputs(outputs);

        auto generatedOutputs = generator_->ProcessThreaded(generatorInputs);

        // Want a value of 0.0 (because it's not from the training set)
        idealOutputs[0] = 0.0;

        brain_->BackPropagationCrossEntropy(generatedOutputs, idealOutputs);

        // Train 100 samples on discriminator then 100 on generator
        if (i > 0u && i % 100u == 0u) {
          // TODO: take the loss function before backpropping this case as not
          // a sample -> is this right?
          for (std::size_t j = 0u; j < 100u; ++j) {
            generatorInputs = GeneratorInputs(outputs);
            generatedOutputs = generator_->ProcessThreaded(generatorInputs);

            idealOutputs[0] = 1.0;

            generator_->BackPropagationCrossEntropy(*brain_.get(),
              generatedOutputs, idealOutputs);
          }
        }

        if (progress > one_hundredth) {
          progress = 0u;
          percent++;
          std::cout << "\rTraining....." << percent << "% ["
            << trainingTimer.ElapsedMicroseconds() << "]";
          trainingTimer.Reset();
        }
      }

      std::cout << "\rTraining.....done\n";
    };

    TrainNeuralNet(brain_.get(), TrainFunction,
      SteppingLearningRate{0.01, 0.01}, 1);
  }

  void TrainOne(const std::size_t i) {
    const auto & normalisedImages = trainingImages_.NormalisedData();
    const std::size_t dimension = trainingImages_.Width();
    const std::size_t inputSize = dimension * dimension;

    std::vector<double> inputs(inputSize);

    const std::size_t length = normalisedImages.size();
    const std::size_t one_hundredth = length / 100u;
    std::size_t progress = i % one_hundredth;
    std::size_t percent = i / one_hundredth;

    std::cout << "\rTraining....." << percent << "%";

    Aligned32ByteRAIIStorage<double> idealOutputs(1u);
    Aligned32ByteRAIIStorage<double> loss;

    const auto & data = trainingOutput_.Data();

    ++progress;

    const auto & normalisedImage = normalisedImages[i];
    inputs.assign(
      normalisedImage.inputs.get(),
      normalisedImage.inputs.get() + inputSize);

    // Want a value of 1.0 (which says it is a digit
    // with a probability of 100%)
    idealOutputs[0] = 1.0;

    brain_->BackPropagationCrossEntropy(inputs, idealOutputs);

    const auto & outputs = data[i];
    std::vector<double> generatorInputs = GeneratorInputs(outputs);

    auto generatedOutputs = generator_->ProcessThreaded(generatorInputs);

    // Want a value of 0.0 (because it's not from the training set)
    idealOutputs[0] = 0.0;

    brain_->BackPropagationCrossEntropy(generatedOutputs, idealOutputs);

    // TODO: take the loss function before backpropping this case as not
    // a sample -> is this right?
    generatorInputs = GeneratorInputs(outputs);
    generatedOutputs = generator_->ProcessThreaded(generatorInputs);

    idealOutputs[0] = 1.0;

    generator_->BackPropagationCrossEntropy(*brain_.get(),
      generatedOutputs, idealOutputs);

    if (progress > one_hundredth) {
      progress = 0u;
      percent++;
      std::cout << "\rTraining....." << percent << "%";
    }
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

    for (std::size_t i = 0; i < size && row < perRow; ++i) {
      double x = -1.0 + (cellDistanceX * column++);
      double y = -1.0 + (cellDistanceY * row);

      if (column == perRow) {
        column = 0;
        row++;
      }

      std::unique_ptr<double[]> normalizedData(new double[normalizedSize]);
      std::unique_ptr<uint8_t[]> data(new uint8_t[pixels]);
      std::size_t dataIndex = 0;

      if (useGenerated) {
        std::vector<double> generatorInputs = GeneratorInputs(
            classifyOutput_.Data()[i]);

        auto outputs = generator_->ProcessThreaded(generatorInputs);

        for (std::size_t j = 0, length = outputs.size(); j < length; ++j) {
          data[dataIndex++] = (uint8_t)(outputs[j] * 255.0);
          data[dataIndex++] = (uint8_t)(outputs[j] * 255.0);
          data[dataIndex++] = (uint8_t)(outputs[j] * 255.0);
          // TODO: image orientation doesn't match outputs? [see other case]
          normalizedData[j] = outputs[j];
        }
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
      const std::size_t inputSize = dimension * dimension;

      std::vector<double> inputs(
        classifyData_[i].get(),
        classifyData_[i].get() + inputSize);

      auto outputs = brain_->Process(inputs);
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
  std::size_t generation_;
  std::unique_ptr<NeuralNet> brain_;
  std::unique_ptr<NeuralNet> generator_;
  ImageFile trainingImages_;
  LabelFile trainingOutput_;
  ImageFile classifyImages_;
  LabelFile classifyOutput_;
  std::vector<std::unique_ptr<double[]>> classifyData_;
  std::size_t classifyIndex_ = 0u;
  bool recreateScene_ = false;

  std::size_t trainingCursor_ = 0u;
};

}
