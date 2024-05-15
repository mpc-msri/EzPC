#pragma once
#include <iostream>

inline void assert_failed(const char* file, int line, const char* function, const char* expression) {
    std::cerr << "Assertion failed: " << expression << " in " << function << " at " << file << ":" << line << std::endl;
    exit(1);
}

#define always_assert(expr) (static_cast <bool> (expr) ? void (0) : assert_failed (__FILE__, __LINE__, __PRETTY_FUNCTION__, #expr))
