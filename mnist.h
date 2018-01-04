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

  void Confidence(double confidence) {
    std::size_t pixels = size_ * size_;
    for (std::size_t i = 0u; i < pixels; ++i) {
      data_[i * 3 + 1] = std::max(data_[i * 3 + 1],
        (uint8_t)(confidence * 200.0));
    }
  }

protected:
  double position_[2];
  std::unique_ptr<uint8_t[]> data_;
  std::size_t size_;
};

}
