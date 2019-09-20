#pragma once

#define NOMINMAX

#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <vector>
#include "macros.h"
#include "threadpool.h"

class Scene;

class ScopedHDC {
public:
  ScopedHDC(::HWND hwnd) : hwnd_(hwnd) {
    hdc_ = ::GetDC(hwnd);
  }

  ~ScopedHDC() {
    ::ReleaseDC(hwnd_, hdc_);
  }

  ScopedHDC(ScopedHDC &&) = delete;
  ScopedHDC(const ScopedHDC &) = delete;

  ScopedHDC operator=(ScopedHDC &&) = delete;
  ScopedHDC operator=(const ScopedHDC &) = delete;

  operator ::HDC() const { return hdc_; }

private:
  ::HWND hwnd_;
  ::HDC hdc_;
};

class OpenGLContext {
public:
  void Create(HWND hwnd) {
    CHECK(!hwnd_);
    hwnd_ = hwnd;
    hdc_ = ::GetDC(hwnd);

    CreatePixelFormat();

    hglrc_ = ::wglCreateContext(hdc_);
    wglMakeCurrent(hdc_, hglrc_);

    InitialiseGL();
  }

  void Destroy() {
    if (hglrc_) {
      wglDeleteContext(hglrc_);
      hglrc_ = nullptr;
    }

    if (hdc_) {
      ::ReleaseDC(hwnd_, hdc_);
      hdc_ = nullptr;
      hwnd_ = nullptr;
    }

    for (auto && listener : destroyListeners_)
      listener();
  }

  void MakeActive() {
    wglMakeCurrent(hdc_, hglrc_);
  }

  void Resize();
  void KeyDown(int key);

  void AddResizeListener(std::function<void()> listener) {
    resizeListeners_.push_back(listener);
  }

  void AddKeyListener(std::function<void(int)> listener) {
    keyListeners_.push_back(listener);
  }

  void AddDestroyListener(std::function<void()> listener) {
    destroyListeners_.push_back(listener);
  }

  void SwapBuffers() {
    CHECK(hdc_);
    ::SwapBuffers(hdc_);
  }

  std::size_t Width() const { return width_; }
  std::size_t Height() const { return height_; }
  ::HWND Handle() { return hwnd_; }

private:
  void CreatePixelFormat() {
    PIXELFORMATDESCRIPTOR pixelFormat;
    ::ZeroMemory(&pixelFormat, sizeof(PIXELFORMATDESCRIPTOR));
    pixelFormat.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pixelFormat.nVersion = 1;
    pixelFormat.dwFlags =
      PFD_DRAW_TO_WINDOW |
      PFD_SUPPORT_OPENGL |
      PFD_DOUBLEBUFFER;
    pixelFormat.dwLayerMask = PFD_MAIN_PLANE;
    pixelFormat.iPixelType = PFD_TYPE_RGBA;
    pixelFormat.cColorBits = 32;
    pixelFormat.cDepthBits = 32;
    pixelFormat.cAccumBits = 0;
    pixelFormat.cStencilBits = 0;

    int format = ::ChoosePixelFormat(hdc_, &pixelFormat);
    CHECK(format != 0);

    ::BOOL result = ::SetPixelFormat(hdc_, format, &pixelFormat);
    CHECK(result != FALSE);
  }

  void InitialiseGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    Resize();
  }

private:
  ::HWND hwnd_ = nullptr;
  ::HDC hdc_ = nullptr;
  ::HGLRC hglrc_ = nullptr;
  std::size_t width_ = 0u;
  std::size_t height_ = 0u;
  std::vector<std::function<void()>> resizeListeners_;
  std::vector<std::function<void(int)>> keyListeners_;
  std::vector<std::function<void()>> destroyListeners_;
};

class ISceneObject {
public:
  virtual void Draw() const = 0;
  virtual void Update(uint64_t ms) = 0;
};

class Scene {
public:
  void Render(OpenGLContext & context) const {
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto && object : objects_)
      object->Draw();

    context.SwapBuffers();
  }

  void Update(uint64_t ms) {
    for (auto && object : objects_)
      object->Update(ms);
  }

  void AddObject(ISceneObject * object) {
    objects_.push_back(object);
  }

  void RemoveObject(ISceneObject * object) {
    objects_.erase(std::find(objects_.begin(), objects_.end(), object));
  }

private:
  std::vector<ISceneObject*> objects_;
};
