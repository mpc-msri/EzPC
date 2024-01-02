#include "../layers.h"
#include "../softmax.h"

void branching_test() {
    const u64 scale = 1;
    std::cout << "> Scale = " << scale << std::endl;
    std::cout << "> Probablistic = " << (ClearText<i64>::probablistic ? "true" : "false") << std::endl;
    srand(time(NULL));
    
    auto fc = new FC<i64, scale>(5, 5);
    fc->weight.fill(1 * (1ULL << scale));
    fc->bias.fill(1 * (1ULL << (2*scale)));
    auto model = Sequential<i64>({
        new BranchAdd<i64>(
            new Sequential<i64> ({
                fc,
                new Truncate<i64>(scale),
            }, false),
            new Identity<i64>()
        ),
    });

    Tensor4D<i64> x(1, 5, 1, 1);
    x.fill(1 * (1ULL << scale));

    model.forward(x);
    model.activation.print();
    model.backward(x);
    // fc->weight.print();
    model.inputDerivative.print();
}

void bn_float() {
    auto bn = new BatchNorm2d<double, 0>(3);
    auto conv = new Conv2D<double, 0>(3, 3, 3, 1);
    conv->filter.fill(1);
    conv->bias.fill(0);
    auto model = Sequential<double>({
        conv,
        bn,
        new ReLU<double>(),
        new Flatten<double>(),
    });

    Tensor4D<double> trainImage(3, 1, 2, 3);
    for(int i = 0; i < trainImage.d1; ++i) {
        for(int j = 0; j < trainImage.d2; ++j) {
            for(int k = 0; k < trainImage.d3; ++k) {
                for(int l = 0; l < trainImage.d4; ++l) {
                    trainImage(i, j, k, l) = (i * trainImage.d2 * trainImage.d3 * trainImage.d4 + j * trainImage.d3 * trainImage.d4 + k * trainImage.d4 + l);
                }
            }
        }
    }
    Tensor4D<double> e(3, 6, 1, 1);
    int numIters = 100;
    for (int i = 0; i < numIters; ++i) {
        model.forward(trainImage);
        softmax<double, 0>(model.activation, e);
        for(u64 b = 0; b < 3; ++b) {
            e(b, 0, 0, 0) -= (1.0/3);
        }
        model.backward(e);
    }
    // bn->gamma.print();
    for(int i = 0; i < bn->gamma.size; ++i) {
        std::cout << (i64)(bn->gamma(i) * (1LL<<24)) << " ";
    }
    std::cout << std::endl;
    // bn->beta.print();
    for(int i = 0; i < bn->beta.size; ++i) {
        std::cout << (i64)(bn->beta(i) * (1LL<<48)) << " ";
    }
    std::cout << std::endl;
    // model.activation.print();
    // bn->inputDerivative.print();
    // model.forward(trainImage, false);
    // model.activation.print();
}


void bn_int() {
    std::cerr << "BN INT" << std::endl;
    const u64 scale = 24;
    auto bn = new BatchNorm2d<i64, scale>(3);
    auto conv = new Conv2D<i64, scale>(3, 3, 3, 1);
    conv->filter.fill(1LL << scale);
    conv->bias.fill(0);
    auto model = Sequential<i64>({
        conv,
        new Truncate<i64>(scale),
        bn,
        new ReLU<i64>(),
        new Flatten<i64>(),
    });

    Tensor4D<i64> trainImage(3, 1, 2, 3);
    for(int i = 0; i < trainImage.d1; ++i) {
        for(int j = 0; j < trainImage.d2; ++j) {
            for(int k = 0; k < trainImage.d3; ++k) {
                for(int l = 0; l < trainImage.d4; ++l) {
                    trainImage(i, j, k, l) = (i * trainImage.d2 * trainImage.d3 * trainImage.d4 + j * trainImage.d3 * trainImage.d4 + k * trainImage.d4 + l) << scale;
                }
            }
        }
    }
    Tensor4D<i64> e(3, 6, 1, 1);
    int numIters = 100;
    for (int i = 0; i < numIters; ++i) {
        model.forward(trainImage);
        softmax<i64, scale>(model.activation, e);
        for(u64 b = 0; b < 3; ++b) {
            e(b, 0, 0, 0) -= ((1LL<<scale)/3);
        }
        model.backward(e);
    }
    bn->gamma.print();
    bn->beta.print();
    // bn->gamma.print(scale);
    // bn->beta.print(2*scale);
    // model.activation.print(scale);
    // bn->inputDerivative.print(scale);
    // model.forward(trainImage, false);
    // model.activation.print(scale);
}
