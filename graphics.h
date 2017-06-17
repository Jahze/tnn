#pragma once

#define NOMINMAX

#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "macros.h"

class Scene;

class OpenGLContext {
public:
  void Create(HWND hwnd) {
    CHECK(!hwnd_);
    hwnd_ = hwnd;
    hdc_ = ::GetDC(hwnd);

    CreatePixelFormat();

    hglrc_ = ::wglCreateContext(hdc_);
    wglMakeCurrent(hdc_, hglrc_);

    ::RECT rect;
    ::GetClientRect(hwnd, &rect);

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
  }

  void Resize();

  void SwapBuffers() {
    CHECK(hdc_);
    ::SwapBuffers(hdc_);
  }

private:
  void CreatePixelFormat() {
    PIXELFORMATDESCRIPTOR pixelFormat;

    pixelFormat.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pixelFormat.nVersion = 1;
    pixelFormat.dwFlags =
      PFD_DRAW_TO_WINDOW |
      PFD_SUPPORT_OPENGL |
      PFD_DOUBLEBUFFER;
    pixelFormat.dwLayerMask = PFD_MAIN_PLANE;
    pixelFormat.iPixelType = PFD_TYPE_COLORINDEX;
    pixelFormat.cColorBits = 8;
    pixelFormat.cDepthBits = 16;
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
};

class ISceneObject {
public:
  virtual void Draw() const = 0;
  virtual void Update(uint64_t ms) = 0;
};

class WorkerThread {
public:
  WorkerThread() {
    thread_ = std::thread(&WorkerThread::Work, this);
  }
  ~WorkerThread() {
    {
      std::unique_lock<std::mutex> lock(lock_);
      running_ = false;
      jobWaiting_.notify_one();
    }
    thread_.join();
  }

  void DoJob(std::function<void(void)> job) {
    CHECK(!job_);
    std::unique_lock<std::mutex> lock(lock_);
    job_ = std::make_unique<std::function<void(void)>>(job);
    jobDone_ = false;
    jobWaiting_.notify_one();
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(lock_);
    while (!jobDone_)
      jobWaiting_.wait(lock);
  }

private:
  void Work() {
    while (running_) {
      std::function<void(void)> job;

      {
        std::unique_lock<std::mutex> lock(lock_);

        while (running_ && !job_)
          jobWaiting_.wait(lock);

        if (!running_)
          return;

        job = *job_;
        job_.release();
      }

      job();

      {
        std::unique_lock<std::mutex> lock(lock_);
        jobWaiting_.notify_one();
        jobDone_ = true;
      }
    }
  }

private:
  std::mutex lock_;
  std::thread thread_;
  bool running_ = true;
  bool jobDone_ = false;
  std::condition_variable jobWaiting_;
  std::unique_ptr<std::function<void(void)>> job_;
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
    //for (auto && object : objects_)
    //  object->Update(ms);

    //const std::size_t size = objects_.size();
    //std::thread thread(&Scene::UpdatePortion,
    //  this, ms, 0, size / 2);

    //for (std::size_t i = size / 2; i < size; ++i)
    //  objects_[i]->Update(ms);

    //thread.join();

    static WorkerThread thread[3]; // TODO: gulp

    const std::size_t size = objects_.size();

    thread[0].DoJob(std::bind(&Scene::UpdatePortion,
      this, ms, 0, size / 4));

    thread[1].DoJob(std::bind(&Scene::UpdatePortion,
      this, ms, size / 4, size / 2));

    thread[2].DoJob(std::bind(&Scene::UpdatePortion,
      this, ms, size / 2, size / 2 + size / 4));

    for (std::size_t i = size / 2 + size / 4; i < size; ++i)
      objects_[i]->Update(ms);

    thread[0].Wait();
    thread[1].Wait();
    thread[2].Wait();
  }

  void AddObject(ISceneObject * object) {
    objects_.push_back(object);
  }

  void RemoveObject(ISceneObject * object) {
    objects_.erase(std::find(objects_.begin(), objects_.end(), object));
  }

private:
  void UpdatePortion(uint64_t ms, std::size_t first, std::size_t last) {
    for (std::size_t i = first; i < last; ++i)
      objects_[i]->Update(ms);
  }

private:
  std::vector<ISceneObject*> objects_;
};
