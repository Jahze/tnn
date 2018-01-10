#define NOMINMAX

#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <chrono>
#include <random>
#include <string>
#include "backprop.h"
#include "chasers.h"
#include "colours.h"
#include "finders.h"
#include "gan.h"
#include "graph.h"
#include "graphics.h"
#include "mnist.h"
#include "mnist_classifier.h"
#include "neural_net.h"
#include "threadpool.h"

OpenGLContext glContext;

bool g_render = true;

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

  case WM_KEYDOWN:
    if (wparam == VK_RETURN) {
      g_render = !g_render;
    }
    return 1;

  default:
    return static_cast<LONG>(::DefWindowProc(hwnd, umsg, wparam, lparam));
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

const std::size_t kFramerate = 60u;
const std::size_t kMsPerFrame = 1000u / kFramerate;
const std::size_t kMsPerGeneration = 1000u * 2u;
const std::size_t kGoals = 3u;

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

#if 0
  finders::Simulation population(
    kMsPerFrame,
    kMsPerGeneration,
    glContext,
    kGoals);
#elif 0
  chasers::Simulation population(
    kMsPerFrame,
    kMsPerGeneration * 5,
    glContext);
#elif 0
  chasers::BackPropSimulation population(
    kMsPerFrame,
    kMsPerGeneration * 15,
    glContext);
#elif 1
  mnist::Classifier population(
    kMsPerFrame,
    glContext,
    "data\\train-images.idx3-ubyte",
    "data\\train-labels.idx1-ubyte",
    "data\\t10k-images.idx3-ubyte",
    "data\\t10k-labels.idx1-ubyte");
#elif 0
  mnist::GAN population(
    kMsPerFrame,
    glContext,
    "data\\train-images.idx3-ubyte",
    "data\\t10k-images.idx3-ubyte");
#elif 0
  backprop::Simulation population(
    kMsPerFrame,
    kMsPerFrame * 2,
    glContext,
    128);
#else
  colours::Simulation population(
    kMsPerFrame,
    kMsPerFrame * 2,
    glContext,
    128,
    100);
#endif

  ::ShowWindow(hwnd, SW_SHOWNORMAL);
  ::UpdateWindow(hwnd);

  //Graph::Limits limits = {-1., 1., 0, 1.};
  //Graph graph("graph", limits);

  //Graph::Series series = {
  //  { {0.0, 0.0}, {1.0, 1.0} }
  //};

  //graph.AddSeries(series);

  ::MSG msg;

  population.Start();

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

    population.Update(g_render);
  }
}
