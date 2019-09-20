#pragma once

#define NOMINMAX

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>
#include <Windows.h>
#include "graphics.h"
#include "macros.h"

class Pen {
public:
  Pen(::HWND hwnd, int r, int g, int b)
      : hwnd_(hwnd), hdc_(hwnd) {
    ::COLORREF colour = RGB(r, g, b);
    pen_ = ::CreatePen(PS_SOLID, 2, colour);
    oldPen_ = (::HPEN)::SelectObject(hdc_, pen_);
  }

  ~Pen() {
    ::SelectObject(hdc_, oldPen_);
    ::DeleteObject(pen_);
  }

  void MoveTo(int x, int y) {
    ::MoveToEx(hdc_, x, y, nullptr);
  }

  void LineTo(int x, int y) {
    ::LineTo(hdc_, x, y);
  }

private:
  ::HWND hwnd_;
  ScopedHDC hdc_;
  ::HPEN pen_;
  ::HPEN oldPen_;
};

class Graph {
public:
  struct Limits {
    double xmin;
    double xmax;
    double ymin;
    double ymax;
  };

  struct Point {
    double x;
    double y;
  };

  struct Series {
    std::vector<Point> points;
    int r = 255;
    int g = 0;
    int b = 0;
  };

  Graph(::HWND hwnd, const Limits & limits)
    : hwnd_(hwnd), limits_(limits) {

    CHECK(hwnd_);
  }

  void AddSeries(const Series & series) {
    series_.push_back(series);
    SignalRedraw();
  }

  void SignalRedraw() {
    ::RECT rect;
    ::GetClientRect(hwnd_, &rect);
    ::InvalidateRect(hwnd_, &rect, TRUE);
  }

  void DrawAxes() {
    ScopedHDC hdc(hwnd_);

    RECT rect;
    ::GetClientRect(hwnd_, &rect);

    DrawLabels(hdc, rect);
    DrawAxesLines(hdc, rect);
  }

  void DrawSeries() {
    if (series_.empty()) return;

    auto Sort = [](const Point & lhs, const Point & rhs) {
      return lhs.x < rhs.x;
    };

    for (auto && series : series_) {
      std::sort(series.points.begin(), series.points.end(), Sort);

      ::RECT rect;
      ::GetClientRect(hwnd_, &rect);

      auto point = ToScreen(rect, series.points[0]);

      Pen pen(hwnd_, series.r, series.g, series.b);
      pen.MoveTo(point.first, point.second);

      const auto size = series.points.size();
      for (std::size_t i = 1; i < size; ++i) {
        point = ToScreen(rect, series.points[i]);
        pen.LineTo(point.first, point.second);
      }
    }
  }

  void Clear() {
    series_.clear();
  }

private:
  const static int AxisBorderSize = 20;

  void DrawLabels(::HDC hdc, const ::RECT & rect) {
    std::stringstream ss;
    ss << std::setprecision(2);

    ss << limits_.ymax;
    std::string ymax = ss.str();
    ss = std::stringstream();
    ::TextOut(hdc, 5, 0, ymax.c_str(), (int)ymax.size());

    ss << limits_.ymin;
    std::string ymin = ss.str();
    ss = std::stringstream();
    ::TextOut(hdc, 5, rect.bottom - 25, ymin.c_str(), (int)ymin.size());

    ss << limits_.xmin;
    std::string xmin = ss.str();
    ss = std::stringstream();
    ::TextOut(hdc, 15, rect.bottom - 15, xmin.c_str(), (int)xmin.size());

    ss << limits_.xmax;
    std::string xmax = ss.str();
    ::TextOut(hdc, rect.right - 10, rect.bottom - 15,
      xmax.c_str(), (int)xmax.size());
  }

  void DrawAxesLines(::HDC hdc, const ::RECT & rect) {
    Pen pen(hwnd_, 0, 0, 0);

    double total = limits_.xmax - limits_.xmin;
    double ratio = -limits_.xmin / total;
    int left = std::max(AxisBorderSize,
      std::min((int)(rect.right - 5),
        (int)(AxisBorderSize + (rect.right - AxisBorderSize - 5)*ratio)));

    pen.MoveTo(left, 5);
    pen.LineTo(left, rect.bottom - AxisBorderSize);

    total = limits_.ymax - limits_.ymin;
    ratio = 1.0 + (limits_.ymin / total);
    int bottom = std::max(5,
      std::min((int)(rect.bottom - AxisBorderSize),
        (int)(5 + (rect.bottom - AxisBorderSize - 5)*ratio)));

    pen.MoveTo(AxisBorderSize, bottom);
    pen.LineTo(rect.right - 5, bottom);
  }

  std::pair<int,int> ToScreen(const ::RECT & rect, const Point & point) {
    double xrange = limits_.xmax - limits_.xmin;
    double x = point.x - (double)limits_.xmin;
    x /= xrange;
    x *= rect.right - AxisBorderSize - 5;
    x += AxisBorderSize;

    double yrange = limits_.ymax - limits_.ymin;
    double y = point.y - (double)limits_.ymin;
    y /= yrange;
    y = (1.0 - y) * (rect.bottom - AxisBorderSize - 5);
    y += 5;

    return { (int)std::round(x), (int)std::round(y) };
  }

private:
  const std::string name_;
  ::HWND hwnd_;
  Limits limits_;
  std::vector<Series> series_;
};

class GraphWindow {
public:
  GraphWindow(const Graph::Limits & limits) {
    static const char WindowClassName[] = "graph window class";
    static const int WindowWidth = 600;
    static const int WindowHeight = 600;

    ::WNDCLASSEX windowClass = CreateWindowClass(WindowClassName);

    if (!::RegisterClassEx(&windowClass))
      LOG("[error] failed to register window class");

    hwnd_ = ::CreateWindow(WindowClassName,
      "graph",
      WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT,
      CW_USEDEFAULT,
      WindowWidth,
      WindowHeight,
      NULL,
      NULL,
      ::GetModuleHandle(NULL),
      NULL);

    if (!hwnd_)
      LOG("[error] failed to create window");

    context_.Create(hwnd_);

    ::SetWindowLongPtr(hwnd_, GWLP_USERDATA, (LONG_PTR)this);
    ::ShowWindow(hwnd_, SW_SHOWNORMAL);
    ::UpdateWindow(hwnd_);

    graph_ = std::make_unique<::Graph>(hwnd_, limits);
  }

  ::Graph * Graph() { return graph_.get(); }
  OpenGLContext * Context() { return &context_; }

private:
  ::WNDCLASSEX CreateWindowClass(const char * name) {
    ::WNDCLASSEX windowClass;
    std::memset(&windowClass, 0, sizeof(windowClass));

    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = (WNDPROC)WindowProc;
    windowClass.hInstance = ::GetModuleHandle(NULL);
    windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    windowClass.lpszMenuName = NULL;
    windowClass.lpszClassName = name;

    return windowClass;
  }

  static LONG WINAPI WindowProc(HWND hwnd,
    UINT umsg,
    WPARAM wparam,
    LPARAM lparam) {

    GraphWindow * window =
      (GraphWindow*)::GetWindowLongPtr(hwnd, GWLP_USERDATA);

    switch (umsg) {
    case WM_PAINT:
    {
      PAINTSTRUCT paint;
      ::BeginPaint(hwnd, &paint);
      ::EndPaint(hwnd, &paint);
      return 1;
    }

    case WM_SIZE:
      window->Context()->Resize();
      break;

    case WM_DESTROY:
      window->Context()->Destroy();
      PostQuitMessage(0);
      return 1;

    default:
      return static_cast<LONG>(::DefWindowProc(hwnd, umsg, wparam, lparam));
    }

    return 0;
  }

private:
  ::HWND hwnd_;
  OpenGLContext context_;
  std::unique_ptr<::Graph> graph_;
};
