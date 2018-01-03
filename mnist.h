#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "macros.h"
#include "neural_net.h"
#include "simulation.h"

namespace mnist {

inline uint32_t Read32Bits(std::ifstream & stream) {
  char buff[4];
  stream.read(buff, 4);
  std::swap(buff[0], buff[3]);
  std::swap(buff[1], buff[2]);
  return *((uint32_t*)buff);
}

struct ImageData {
  std::unique_ptr<uint8_t[]> bytes;

  ImageData(std::size_t size) {
    bytes.reset(new uint8_t [size]);
  }
};

struct NormalisedImageData {
  std::unique_ptr<double[]> inputs;

  NormalisedImageData(std::size_t size, const ImageData & data) {
    inputs.reset(new double[size]);
    for (std::size_t i = 0u; i < size; ++i)
      inputs[i] = static_cast<double>(data.bytes[i]) / 255.0;
  }
};

class ImageFile {
public:
  static ImageFile Read(const std::string & filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return ImageFile();

    uint32_t magicNumber = Read32Bits(file);
    CHECK(magicNumber == 0x803);

    ImageFile imageFile;
    imageFile.size_ = Read32Bits(file);
    imageFile.height_ = Read32Bits(file);
    imageFile.width_ = Read32Bits(file);

    const std::size_t imageSize = imageFile.height_ * imageFile.width_;

    for (std::size_t i = 0; i < imageFile.size_; ++i)
      imageFile.images_.emplace_back(imageSize);

    const std::size_t totalBytes = imageSize * imageFile.size_;
    std::size_t image = 0;

    for (std::size_t i = 0; i < totalBytes; ++i) {
      std::size_t remainder = i % imageSize;
      file.read((char*)&imageFile.images_[image].bytes[remainder], 1);
      if (remainder == imageSize - 1) {
        imageFile.normalisedImages_.emplace_back(
          imageSize, imageFile.images_[image]);
        ++image;
      }
    }

    imageFile.valid_ = true;
    return imageFile;
  }

  bool Valid() const { return valid_; }
  std::size_t Size() const { return size_; }
  std::size_t Width() const { return width_; }
  std::size_t Height() const { return height_; }

  const std::vector<ImageData> & Data() const { return images_; }

  const std::vector<NormalisedImageData> & NormalisedData() const {
    return normalisedImages_;
  }

private:
  bool valid_ = false;
  std::size_t size_ = 0u;
  std::size_t width_ = 0u;
  std::size_t height_ = 0u;
  std::vector<ImageData> images_;
  std::vector<NormalisedImageData> normalisedImages_;
};

class LabelFile {
public:
  static LabelFile Read(const std::string & filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return LabelFile();

    uint32_t magicNumber = Read32Bits(file);
    CHECK(magicNumber == 0x801);

    std::size_t items = Read32Bits(file);

    LabelFile labelFile;
    for (std::size_t i = 0u; i < items; ++i) {
      char label;
      file.read(&label, 1);

      std::vector<double> outputs(10, 0.0);
      outputs[label] = 1.0;

      labelFile.outputs_.emplace_back(std::move(outputs));
    }

    return labelFile;
  }

  using LabelData = std::vector<double>;

  std::size_t GetDigit(std::size_t index) const {
    const auto & data = outputs_[index];

    for (std::size_t i = 0u; i < 10; ++i) {
      if (data[i] > 0.0) return i;
    }

    return 0u;
  }

  const std::size_t Size() const { return outputs_.size(); }
  const std::vector<LabelData> & Data() const { return outputs_; }

private:
  std::vector<LabelData> outputs_;
};

class Glyph : public ISceneObject {
public:
  Glyph(double x, double y, uint8_t * data, std::size_t size) {
    position_[0] = x;
    position_[1] = y;
    size_ = size;
    data_.reset(new uint8_t [size* size * 3]);
    std::memcpy(data_.get(), data, size * size * 3);
  }

  void Draw() const override {
    glRasterPos2d(position_[0], position_[1]);
    glDrawPixels((GLsizei)size_, (GLsizei)size_,
      GL_RGB, GL_UNSIGNED_BYTE, data_.get());
  }

  void Update(uint64_t ms) override {
  }

  void Matched() {
    std::size_t pixels = size_ * size_;
    for (std::size_t i = 0u; i < pixels; ++i)
      data_[i * 3 + 1] = 200;
  }

  void NotMatched() {
    std::size_t pixels = size_ * size_;
    for (std::size_t i = 0u; i < pixels; ++i)
      data_[i * 3] = 200;
  }

protected:
  double position_[2];
  std::unique_ptr<uint8_t[]> data_;
  std::size_t size_;
};

class Simulation : public ::SimpleSimulation {
public:
  Simulation(std::size_t msPerFrame,
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

    for (int i = 0; i < 1; ++i)
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

    Timer trainingTimer;

    for (std::size_t i = 0u; i < length; ++i, ++progress) {
      const auto & normalisedImage = normalisedImages[i];

      inputs.assign(
        normalisedImage.inputs.get(),
        normalisedImage.inputs.get() + inputSize);

      const auto & outputs = data[i];

      auto lossFunction = [&outputs](double value, std::size_t idx) {
        return -(outputs[idx] - value);
      };

      brain_->BackPropagationThreaded(inputs, lossFunction);

      if (progress > one_hundredth) {
        progress = 0u;
        percent++;
        std::cout << "\rTraining....." << percent << "% ["
          << trainingTimer.ElapsedMicroseconds() << "]";
        trainingTimer.Reset();
      }
    }

    std::cout << "\rTraining.....done\n";
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
