#define NOMINMAX

#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <chrono>
#include <random>
#include <string>
#include "finders.h"
#include "graphics.h"
#include "neural_net.h"

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
const std::size_t kMsPerGeneration = 1000u * 5u;
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

  std::cout << "Generation 0\n";

  FinderPopulation population(
    kMsPerFrame,
    kMsPerGeneration,
    glContext,
    kGoals);

  ::ShowWindow(hwnd, SW_SHOWNORMAL);
  ::UpdateWindow(hwnd);

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
