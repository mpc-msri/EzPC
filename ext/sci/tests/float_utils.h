#include <iostream>
#include <random>

using namespace std;

float sample_float(std::mt19937& generator, float lb, float ub) {
  float f;
  do {
    uint32_t fint = generator();
    f = *((float *) &fint);
  } while ((f < lb || f >= ub) || (!isnormal(f) && (f != 0.0)));
  return f;
}
