#pragma once

#define NOMINMAX

#include <Windows.h>
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
