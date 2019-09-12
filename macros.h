#pragma once

#define NOMINMAX

#include <Windows.h>
#include <cfloat>
#include <iostream>
#include <string>

#define LOG(message) std::cerr << message << "\n"

#define CHECK(assertion) \
  do { \
    if (!(assertion)) { \
      std::string message = "\"" #assertion "\" == false " \
        " at " __FILE__ ":"; \
      message += std::to_string(__LINE__); \
      ::MessageBox(NULL, message.c_str(), "Error", MB_OK); \
      std::exit(1); \
    } \
  } while (false)

#define TWO_PI 6.28318530718

inline void SetFloatingPointExceptionMode() {
#if FP_EXCEPTIONS
  _controlfp(_EM_INEXACT, _MCW_EM & (~_EM_UNDERFLOW));
#endif

  return;
}
