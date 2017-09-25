#include "graphics.h"

void OpenGLContext::Resize() {
  CHECK(hwnd_);

  ::RECT rect;
  ::GetClientRect(hwnd_, &rect);
  glViewport(0, 0, rect.right, rect.bottom);

  glLoadIdentity();
  gluOrtho2D(0, 0, rect.right, rect.bottom);

  width_ = static_cast<std::size_t>(rect.right);
  height_ = static_cast<std::size_t>(rect.bottom);

  for (auto && listener : resizeListeners_)
    listener();
}
