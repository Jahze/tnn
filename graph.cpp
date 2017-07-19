#include <string>
#include "graph.h"
#include "macros.h"

LONG WINAPI GraphWindowProc(HWND hwnd, UINT umsg, WPARAM wparam, LPARAM lparam) {
  switch (umsg) {
  case WM_CREATE:
    return 1;

  case WM_PAINT:
  {
    Graph * graph = (Graph*)::GetWindowLongPtr(hwnd, GWLP_USERDATA);

    ::PAINTSTRUCT ps;
    ::HDC hdc = ::BeginPaint(hwnd, &ps);

    ::FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));

    if (graph) {
      graph->DrawAxes();
      graph->DrawSeries();
    }

    ::EndPaint(hwnd, &ps);
    return 1;
  }

  case WM_SIZE:
  {
    Graph * graph = (Graph*)::GetWindowLongPtr(hwnd, GWLP_USERDATA);
    graph->SignalRedraw();
    return 1;
  }

  case WM_DESTROY:
    PostQuitMessage(0);
    return 1;

  case WM_KEYDOWN:
    return 1;

  default:
    return static_cast<LONG>(::DefWindowProc(hwnd, umsg, wparam, lparam));
  }

  return 0;
}

const char * Graph::WindowClass() {
  static const char ClassName[] = "GraphWindowClass";
  static bool Initialised = false;

  if (!Initialised) {
    Initialised = true;

    ::WNDCLASSEX windowClass;
    std::memset(&windowClass, 0, sizeof(windowClass));

    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = (WNDPROC)GraphWindowProc;
    windowClass.hInstance = ::GetModuleHandle(NULL);
    windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    windowClass.lpszMenuName = NULL;
    windowClass.lpszClassName = ClassName;

    bool result = !!::RegisterClassEx(&windowClass);
    CHECK(result);
  }

  return ClassName;
}
