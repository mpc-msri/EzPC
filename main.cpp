#include <iostream>
#include <vector>
#include "layers.h"

int main() {
    auto model = Sequential<i64>({
        new Flatten<i64>(),
        new FC<i64>(784, 512),
        new ReLUTruncate<i64>(5),
        new FC<i64>(512, 256),
        new ReLUTruncate<i64>(5),
        new FC<i64>(256, 10),
        new Truncate<i64>(5)
    });

    Tensor4D<i64> a(1, 28, 28, 1);
    a.randomize();

    model.forward(a);

    for(u64 i = 0; i < 10; i++) {
        std::cout << model.activation[0][i][0][0] << " ";
    }
}