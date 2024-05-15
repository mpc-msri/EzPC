#include "model_accuracy.h"

int main() {
    omp_set_num_threads(64);
    sytorch_init();
    using T = i64;
    auto res = getLossCIFAR10<T>("CNN3", 24, "output/training/e2e/loss/CNN31e/weights/", 0, 0, 0);
    printf("Accuracy=%lf, Loss=%lf\n", res.first, res.second);
}