#pragma once

#include "graphics.h"
#include "mnist.h"

namespace mnist {

class Classifier : public ::SimpleSimulation {
public:
  Classifier(std::size_t msPerFrame,
             OpenGLContext & context,
             const std::string & trainingFilename,
             const std::string & trainingLabels,
             const std::string & classifyFilename,
             const std::string & classifyLabels)
    : ::SimpleSimulation(msPerFrame)
    , context_(context) {

    scene_.reset(new Scene());

    trainingImages_ = mnist::ImageFile::Read(trainingFilename);

    CHECK(trainingImages_.Size() > 0);
    CHECK(trainingImages_.Width() == trainingImages_.Height());

    trainingOutput_ = mnist::LabelFile::Read(trainingLabels);

    CHECK(trainingOutput_.Size() == trainingImages_.Size());

    const std::size_t dimension = trainingImages_.Width();
    brain_.reset(new NeuralNet(dimension * dimension, 10, 1, 300));

    TrainModel();

    classifyImages_ = mnist::ImageFile::Read(classifyFilename);

    CHECK(classifyImages_.Size() > 0);
    CHECK(classifyImages_.Width() == classifyImages_.Height());
    CHECK(classifyImages_.Width() == trainingImages_.Width());

    classifyOutput_ = mnist::LabelFile::Read(classifyLabels);

    CHECK(classifyOutput_.Size() == classifyImages_.Size());

    Classify();

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

    if (render && classifyIndex_ < objects_.size()) {
      const std::size_t dimension = classifyImages_.Width();
      const std::size_t inputSize = dimension * dimension;

     const auto & classifyData = classifyImages_.NormalisedData();

      std::vector<double> inputs(
        classifyData[classifyIndex_].inputs.get(),
        classifyData[classifyIndex_].inputs.get() + inputSize);

      auto outputs = brain_->Process(inputs);
      std::size_t digit = 0u;
      for (std::size_t i = 0u, length = outputs.size(); i < length; ++i) {
        if (outputs[i] > outputs[digit])
          digit = i;
      }

      if (digit == classifyOutput_.GetDigit(classifyIndex_))
        objects_[classifyIndex_]->Matched();
      else
        objects_[classifyIndex_]->NotMatched();

      //std::cout << classifyIndex_ << ": " << std::setprecision(2) << "{";

      //for (std::size_t i = 0u, length = outputs.size(); i < length; ++i) {
      //  if (digit == i) std::cout << " [" << outputs[i] << "]";
      //  else std::cout << " " << outputs[i];
      //}

      //std::cout << " } " << digit << " <-> "
      //  << classifyOutput_.GetDigit(classifyIndex_) << "\n";

      classifyIndex_++;
    }
    else if (classifyIndex_ == objects_.size()) {
      recreateScene_ = true;
    }

    scene_->Update(ms);

    if (render)
      scene_->Render(context_);
  }

  void TrainModel() {
    auto TrainFunction = [this]() {
      const auto & normalisedImages = trainingImages_.NormalisedData();
      const std::size_t dimension = trainingImages_.Width();
      const std::size_t inputSize = dimension * dimension;

      const auto & data = trainingOutput_.Data();

      std::vector<double> inputs(inputSize);

      std::cout << "Training.....0%";

      const std::size_t length = normalisedImages.size();
      const std::size_t one_hundredth = length / 100u;
      std::size_t progress = 0u;
      std::size_t percent = 0u;

      Timer totalTrainingTimer;
      Timer trainingTimer;

      Aligned32ByteRAIIStorage<double> idealOutputs(10u);

      for (std::size_t i = 0u; i < length; ++i, ++progress) {
        const auto & normalisedImage = normalisedImages[i];

        inputs.assign(
          normalisedImage.inputs.get(),
          normalisedImage.inputs.get() + inputSize);

        const auto & outputs = data[i];
        std::memcpy(idealOutputs.Get(), outputs.data(), 10u * sizeof(double));

        // This is cross-entropy? (derivative might be incorrect)
        // But if ideal value is 0 it says there is no loss?
        //auto lossFunction = [&outputs](double value, std::size_t idx) {
        //  return -(value/outputs[idx]);
        //};

        brain_->BackPropagationThreaded(inputs, idealOutputs);
        if (i > 0 && i % 100u == 0)
          brain_->CommitDeltas();

        if (progress > one_hundredth) {
          progress = 0u;
          percent++;
          std::cout << "\rTraining....." << percent << "% ["
            << trainingTimer.ElapsedMicroseconds() << "]";
          trainingTimer.Reset();
        }
      }

      std::cout << "\rTraining.....done\n";

      std::cout << "Took " << totalTrainingTimer.ElapsedSeconds() << "s\n";
    };

    brain_->SetUpdateType(NeuralNet::UpdateType::Batched);
    brain_->SetLearningRate(0.001);
    TrainFunction();
    //TrainNeuralNet(brain_.get(), TrainFunction,
    //  SteppingLearningRate{0.5, 0.1}, 1);
  }

  void Classify() {
    const auto & classifyData = classifyImages_.NormalisedData();
    const std::size_t length = classifyData.size();

    const std::size_t dimension = classifyImages_.Width();
    const std::size_t inputSize = dimension * dimension;

    std::vector<double> inputs(inputSize);

    std::size_t matched = 0u;
    std::size_t byDigit[10][2];
    for (std::size_t i = 0u; i < 10; ++i)
      byDigit[i][0] = byDigit[i][1] = 0;

    for (std::size_t datum = 0; datum < length; ++datum) {
      inputs.assign(
        classifyData[datum].inputs.get(),
        classifyData[datum].inputs.get() + inputSize);

      auto outputs = brain_->Process(inputs);
      std::size_t digit = 0u;
      for (std::size_t i = 0u, length = outputs.size(); i < length; ++i) {
        if (outputs[i] > outputs[digit])
          digit = i;
      }

      auto expected = classifyOutput_.GetDigit(datum);

      byDigit[expected][0]++;

      if (digit == expected) {
        matched++;
        byDigit[digit][1]++;
      }
    }

    double matchRate =
      static_cast<double>(matched) / static_cast<double>(length);
    matchRate *= 100.0;

    std::cout << "Match rate = " << matchRate << "%\n";

    for (std::size_t i = 0; i < 10; ++i) {
      double digitMatchRate =
        static_cast<double>(byDigit[i][1]) / static_cast<double>(byDigit[i][0]);

      digitMatchRate *= 100.0;

      std::cout << "Match rate for " << i << " = "<< digitMatchRate << "%\n";
    }
  }

  void DestroyScene() {
    classifyIndex_ = 0;
    scene_.reset(new Scene());
    objects_.clear();
  }

  void RenderImages() {
    const auto & images = classifyImages_.Data();
    const std::size_t size = images.size();

    const std::size_t dimension = classifyImages_.Width();
    const std::size_t pixels = 3 * dimension * dimension;

    const std::size_t perRow = context_.Width() / dimension;
    const std::size_t perColumn = context_.Height() / dimension;
    const double cellDistanceX = 2.0 / (double)perRow;
    const double cellDistanceY = 2.0 / (double)perColumn;
    std::size_t row = 0;
    std::size_t column = 0;

    for (std::size_t i = 0; i < size && row < perRow; ++i) {
      double x = -1.0 + (cellDistanceX * column++);
      double y = -1.0 + (cellDistanceY * row);

      if (column == perRow) {
        column = 0;
        row++;
      }

      const auto & image = images[i];

      std::unique_ptr<uint8_t[]> data(new uint8_t[pixels]);
      std::size_t dataIndex = 0;

      for (std::size_t j = 0; j < dimension; ++j) {
        for (std::size_t k = 0; k < dimension; ++k) {
          const std::size_t byte = dimension - j - 1;
          data[dataIndex++] = image.bytes[k + byte * dimension];
          data[dataIndex++] = image.bytes[k + byte * dimension];
          data[dataIndex++] = image.bytes[k + byte * dimension];
        }
      }

      CHECK(dataIndex == pixels);

      objects_.emplace_back(new Glyph(x, y, data.get(), dimension));
      auto & object = objects_.back();
      scene_->AddObject(object.get());
    }
  }

private:
  OpenGLContext & context_;
  std::unique_ptr<Scene> scene_;
  std::vector<std::unique_ptr<Glyph>> objects_;
  std::size_t generation_;
  std::unique_ptr<NeuralNet> brain_;
  ImageFile trainingImages_;
  ImageFile classifyImages_;
  LabelFile trainingOutput_;
  LabelFile classifyOutput_;
  std::size_t classifyIndex_ = 0u;
  bool recreateScene_ = false;
};

}
