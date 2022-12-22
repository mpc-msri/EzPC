#pragma GCC optimize "trapv"
#include <iostream>
#include <cstdint>

int main() {
    uint64_t scale = 12;
    int64_t val = -1;
    std::cout << (val >> scale) << std::endl;

    int64_t a;
    int64_t b;
    std::cin >> a;
    std::cin >> b;
    int64_t c = a + b;
    std::cout << c << std::endl;
}