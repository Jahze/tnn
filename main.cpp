#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <chrono>
#include <random>
#include <string>
#include "graphics.h"
#include "neural_net.h"

OpenGLContext glContext;

float RandomFloat(float min, float max) {
  static std::random_device rd;
  static std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(generator);
}

LONG WINAPI MainWndProc(HWND hwnd, UINT umsg, WPARAM wparam, LPARAM lparam) {
  switch (umsg) {
  case WM_CREATE:
    glContext.Create(hwnd);
    return 1;

  case WM_PAINT:
  {
    PAINTSTRUCT paint;
    ::BeginPaint(hwnd, &paint);
    ::EndPaint(hwnd, &paint);
    return 1;
  }

  case WM_SIZE:
    glContext.Resize();
    break;

  case WM_DESTROY:
    glContext.Destroy();
    PostQuitMessage(0);
    return 1;

  default:
    return ::DefWindowProc(hwnd, umsg, wparam, lparam);
  }

  return 0;
}

::WNDCLASSEX CreateWindowClass(const char * name) {
  ::WNDCLASSEX windowClass;
  std::memset(&windowClass, 0, sizeof(windowClass));

  windowClass.cbSize = sizeof(windowClass);
  windowClass.lpfnWndProc = (WNDPROC)MainWndProc;
  windowClass.hInstance = ::GetModuleHandle(NULL);
  windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  windowClass.lpszMenuName = NULL;
  windowClass.lpszClassName = name;

  return windowClass;
}

void Render(const Scene & scene) {
  glClear(GL_COLOR_BUFFER_BIT);

  scene.Render();

  glContext.SwapBuffers();
}

class SimpleObject : public ISceneObject {
public:
  SimpleObject(float x, float y) {
    position_[0] = x;
    position_[1] = y;
  }

  void SetSize(float size) { size_ = size; }

  void Draw() const override {
    glBegin(GL_QUADS);

      glColor3fv(colour_);
      glVertex2f(position_[0] - size_, position_[1] - size_);
      glVertex2f(position_[0] + size_, position_[1] - size_);
      glVertex2f(position_[0] + size_, position_[1] + size_);
      glVertex2f(position_[0] - size_, position_[1] + size_);

    glEnd();
  }

  void Update(uint64_t ms) override {
  }

protected:
  float colour_[3] = { 1.0f, 1.0f, 1.0f };
  float position_[2];
  float size_ = 0.05f;

  
};

class SmartObject : public SimpleObject {
public:
  SmartObject(float x, float y, float goalx, float goaly)
    : SimpleObject(x, y) {
    goal_[0] = goalx;
    goal_[1] = goaly;
  }

  void Update(uint64_t ms) override {
    static const float kSpeed = 0.01f;

    //float xSpeed = RandomFloat(-1.0f, 1.0f) * kSpeed;
    //float ySpeed = RandomFloat(-1.0f, 1.0f) * kSpeed;

    std::vector<double> inputs{position_[0], position_[1], goal_[0], goal_[1]};
    auto outputs = brain_.Process(inputs);

    //const float xSpeed = (static_cast<float>(outputs[0]) +
    //  (-1.f * static_cast<float>(outputs[1]))) * kSpeed *
    //  static_cast<float>(outputs[2]);
    //const float ySpeed = (static_cast<float>(outputs[3]) +
    //  (-1.f * static_cast<float>(outputs[4]))) * kSpeed *
    //  static_cast<float>(outputs[5]);

    //const float xSpeed = (static_cast<float>(outputs[0]) +
    //  (-1.f * static_cast<float>(outputs[1]))) * kSpeed;
    //const float ySpeed = (static_cast<float>(outputs[3]) +
    //  (-1.f * static_cast<float>(outputs[4]))) * kSpeed;

    float xSpeed = outputs[0] > outputs[1] ? kSpeed : -kSpeed;
    float ySpeed = outputs[2] > outputs[3] ? kSpeed : -kSpeed;

    xSpeed *= static_cast<float>(outputs[4]);
    ySpeed *= static_cast<float>(outputs[5]);

    if (position_[0] + xSpeed - size_ < -1.0f)
      return;

    if (position_[1] + ySpeed - size_ < -1.0f)
      return;

    if (position_[0] + xSpeed + size_ > 1.0f)
      return;

    if (position_[1] + ySpeed + size_ > 1.0f)
      return;

    position_[0] += xSpeed;
    position_[1] += ySpeed;
  }

private:
  float goal_[2];

  const static std::size_t brainInputs = 4;
  const static std::size_t brainOutputs = 6;

  NeuralNet brain_{ brainInputs, brainOutputs, 1, 16 };
};

int main() {
  static const char WindowClassName[] = "tnn window class";
  static const int WindowWidth = 600;
  static const int WindowHeight = 600;

  ::WNDCLASSEX windowClass = CreateWindowClass(WindowClassName);

  if (!::RegisterClassEx(&windowClass)) {
    LOG("[error] failed to register window class");
    return 0;
  }

  ::HWND hwnd = ::CreateWindow(WindowClassName,
    "tnn",
    WS_OVERLAPPEDWINDOW,
    CW_USEDEFAULT,
    CW_USEDEFAULT,
    WindowWidth,
    WindowHeight,
    NULL,
    NULL,
    ::GetModuleHandle(NULL),
    NULL);

  if (!hwnd) {
    LOG("[error] failed to create window");
    return FALSE;
  }

  const std::size_t kPopulationSize = 20u;
  const std::size_t kFramerate = 60u;
  const std::size_t kMsPerFrame = 1000u / kFramerate;

  Scene scene;
  const float goalX = RandomFloat(0.1f, 0.9f);
  const float goalY = RandomFloat(0.1f, 0.9f);
  SimpleObject goal(goalX, goalY);
  goal.SetSize(0.01f);
  scene.AddObject(&goal);
  std::vector<std::unique_ptr<SmartObject>> objects;
  for (std::size_t i = 0; i < kPopulationSize; ++i) {
    objects.emplace_back(new SmartObject(0.0f, 0.0f, goalX, goalY));
    scene.AddObject(objects.back().get());
  }

  ::ShowWindow(hwnd, SW_SHOWNORMAL);
  ::UpdateWindow(hwnd);

  ::MSG msg;

  auto lastTick = std::chrono::high_resolution_clock::now();

  while (true) {
    while (::PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE) == TRUE) {
      if (::GetMessage(&msg, NULL, 0, 0)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
      }
      else {
        return 0;
      }
    }

    Render(scene);

    auto nextTick = std::chrono::high_resolution_clock::now();
    auto elapsed = nextTick - lastTick;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

    if (ms.count() > kMsPerFrame) {
      scene.Update(ms.count());

      lastTick = std::chrono::high_resolution_clock::now();
    }
  }
}
