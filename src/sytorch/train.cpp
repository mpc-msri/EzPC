#include <iomanip>
#include <sytorch/train.h>
#include <sytorch/datasets/cifar10.h>
#include <sytorch/datasets/mnist.h>
#include <sytorch/softmax.h>

void printprogress(double percent) {
    int val = (int) (percent * 100);
    int lpad = (int) (percent * 50);
    int rpad = 50 - lpad;
    std::cout << "\r" << "[" << std::setw(3) << val << "%] ["
              << std::setw(lpad) << std::setfill('=') << ""
              << std::setw(rpad) << std::setfill(' ') << "] ";
    std::cout.flush();
}
