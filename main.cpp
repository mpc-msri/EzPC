#include <iostream>
#include <vector>
#include "layers.h"
#include "mnist.h"
#include <cmath>

int main() {

    const u64 scale = 12;
    const u64 miniBatch = 200;
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

    Tensor4D<i64> e(1, 10, 1, 1);
    for(int epoch = 0; epoch < 20; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        for(u64 i = 0; i < miniBatch; ++i) {
            Tensor4D<i64> trainImage(1, 28, 28, 1);
            // fetch image
            for(u64 j = 0; j < 28; ++j) {
                for(u64 k = 0; k < 28; ++k) {
                    trainImage(0, j, k, 0) = train_image[i][j * 28 + k] * (1ULL << scale);
                }
            }
            model.forward(trainImage);
            i64 max = model.activation(0, 0, 0, 0);
            for(u64 j = 1; j < 10; ++j) {
                if(model.activation(0, j, 0, 0) > max) {
                    max = model.activation(0, j, 0, 0);
                }
            }
            double den = 0.0;
            for(u64 j = 0; j < 10; ++j) {
                e(0, j, 0, 0) = model.activation(0, j, 0, 0) - max;
                double x = ((double)e(0, j, 0, 0)) / (1ULL << scale);
                den += exp(x);
            }

            for(u64 j = 0; j < 10; ++j) {
                double x = ((double)e(0, j, 0, 0)) / (1ULL << scale);
                e(0, j, 0, 0) = (i64)((exp(x) / den) * (1ULL << scale));
            }
            e(0, train_label[i], 0, 0) -= (1ULL << scale);
            model.backward(e);
        }

    }
    
    model.forward(trainSet);

    u64 correct = 0;
    for(u64 i = 0; i < miniBatch; i++) {
        if (model.activation.argmax(i) == train_label[i]) {
            correct++;
        }
    }
    std::cout << "Training Set Accuracy: " << correct << "/" << miniBatch << std::endl;
}