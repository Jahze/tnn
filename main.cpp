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

void Render(const Scene & scene) {
  glClear(GL_COLOR_BUFFER_BIT);

  scene.Render();

  glContext.SwapBuffers();
}

const std::size_t kFramerate = 60u;
const std::size_t kMsPerFrame = 1000u / kFramerate;
const std::size_t kMsPerGeneration = 1000u * 5u;
const std::size_t kGoals = 3u;

Scene * PreEvolve(std::size_t num,
                  Scene * scene,
                  Population & population,
                  std::size_t & generation) {

  for (std::size_t gen = 0; gen < num; ++gen) {
    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t goal = 0; goal < kGoals; ++goal) {
      const uint64_t extraTicks = (kMsPerGeneration * 10) / kMsPerFrame;
      for (uint64_t i = 0; i < extraTicks; ++i)
        scene->Update(kMsPerFrame);

      population.CreateNewGoal();
    }

    scene = population.Evolve();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

    std::cout << "Generation " << ++generation << " [" << ms.count() << "ms]\n";
  }

  return scene;
}

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

  std::size_t generation = 0;
  std::cout << "Generation " << generation << "\n";

  Population population;
  Scene * scene = population.GenerateInitialPopulation();

  scene = PreEvolve(0, scene, population, generation);

  ::ShowWindow(hwnd, SW_SHOWNORMAL);
  ::UpdateWindow(hwnd);

  ::MSG msg;

  auto lastTick = std::chrono::high_resolution_clock::now();
  std::chrono::milliseconds thisSpawn(0u);
  std::size_t lastGoal = 0u;

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

    if (g_render) {
      double fitness = 0.0;
      SmartObject * best = nullptr;
      for (auto && object : population) {
        double f = object->CalculateFitness();
        if (f > fitness) {
          best = object.get();
          fitness = f;
        }
      }

      best->SetColour(1.0, 0.0, 0.0);

      Render(*scene);

      best->SetColour(1.0, 1.0, 1.0);
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - lastTick;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

    if (!g_render) ms = std::chrono::milliseconds(kMsPerFrame+1);

    if (ms.count() > kMsPerFrame) {
      scene->Update(ms.count());

      lastTick = std::chrono::high_resolution_clock::now();
    }

    thisSpawn += ms;

    if (thisSpawn.count() > kMsPerGeneration) {
      // Run it for longer than shown to get to the end
      const uint64_t extraTicks = (kMsPerGeneration * 10) / kMsPerFrame;
      for (uint64_t i = 0; i < extraTicks; ++i)
        scene->Update(kMsPerFrame);

      if (++lastGoal < kGoals) {
        population.CreateNewGoal();
        lastTick = std::chrono::high_resolution_clock::now();
      }
      else {
        scene = population.Evolve();

        std::cout << "Generation " << ++generation << "\n";

        lastGoal = 0u;
      }

      thisSpawn = std::chrono::milliseconds(0u);
      lastTick = std::chrono::high_resolution_clock::now();
    }

  }
}
