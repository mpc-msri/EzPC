#include <Eigen/Dense>
#include <chrono>
#include <iostream>

int main()
{
    int d1 = 5000;
    int d2 = 5000;
    int d3 = 5000;

    uint64_t *a = new uint64_t[d1 * d2];
    uint64_t *b = new uint64_t[d2 * d3];
    uint64_t *c = new uint64_t[d1 * d3];
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a, d1, d2);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b, d2, d3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c, d1, d3);
    auto start = std::chrono::high_resolution_clock::now();
    eC = (eA * eB);//.template triangularView<Eigen::Lower>();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Eigen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}