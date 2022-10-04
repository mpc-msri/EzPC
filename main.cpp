#include <iostream>
#include <vector>
#include "layers.h"
#include "mnist.h"
#include <cmath>

int main() {

    const u64 scale = 12;
    const u64 miniBatch = 1;
    load_mnist();

    Tensor4D<i64> trainSet(miniBatch, 28, 28, 1);

    for(u64 i = 0; i < miniBatch; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                trainSet(i, j, k, 0) = train_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    auto model = Sequential<i64>({
        new Flatten<i64>(),
        // new FC<i64>(784, 512),
        // new ReLUTruncate<i64>(scale),
        // new FC<i64>(512, 256),
        // new ReLUTruncate<i64>(scale),
        new FC<i64>(784, 10),
        new Truncate<i64>(scale)
    });

    Tensor4D<i64> e(miniBatch, 10, 1, 1);
    for(int epoch = 0; epoch < 10; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        model.forward(trainSet);

        for(u64 i = 0; i < miniBatch; ++i) {
            if (i == 0) {
                std::cout << "Actual Label: " << train_label[i] << std::endl;
                std::cout << "FP Scores: ";
                for(u64 j = 0; j < 10; ++j) {
                    std::cout << model.activation(i, j, 0, 0) << " ";
                }
                std::cout << std::endl;
            }
            i64 max = model.activation(i, 0, 0, 0);
            for(u64 j = 1; j < 10; ++j) {
                if(model.activation(i, j, 0, 0) > max) {
                    max = model.activation(i, j, 0, 0);
                }
            }
            double den = 0.0;
            for(u64 j = 0; j < 10; ++j) {
                e(i, j, 0, 0) = model.activation(i, j, 0, 0) - max;
                double x = ((double)e(i, j, 0, 0)) / (1ULL << scale);
                den += exp(x);
            }

            for(u64 j = 0; j < 10; ++j) {
                double x = ((double)e(i, j, 0, 0)) / (1ULL << scale);
                e(i, j, 0, 0) = (i64)((exp(x) / den) * (1ULL << scale));
            }
            e(i, train_label[i], 0, 0) -= (1ULL << scale);
            if (i == 0) {
                std::cout << "FP Softmax Scores: ";
                for(u64 j = 0; j < 10; ++j) {
                    std::cout << e(i, j, 0, 0) << " ";
                }
                std::cout << std::endl;
            }
        }

        model.backward(e);
    }
    
    model.forward(trainSet);

    for(u64 i = 0; i < 10; i++) {
        std::cout << model.activation(0, i, 0, 0) << " ";
    }
    std::cout << std::endl;
    std::cout << "Label: " << train_label[0] << std::endl;
}