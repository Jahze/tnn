#include "graphics.h"

void OpenGLContext::Resize() {
  CHECK(hwnd_);

  ::RECT rect;
  ::GetClientRect(hwnd_, &rect);
  glViewport(0, 0, rect.right, rect.bottom);

  glLoadIdentity();
  gluOrtho2D(0, 0, rect.right, rect.bottom);
}
