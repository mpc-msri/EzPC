#include "layers.h"
#include "train.h"

void lenet_int() {
    const u64 scale = 24;
    auto model = Sequential<i64>({
        new Conv2D<i64, scale>(1, 8, 5),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(2),
        new Conv2D<i64, scale>(8, 16, 5),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(2),
        new Flatten<i64>(),
        new FC<i64, scale>(256, 128),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(128, 10),
        new Truncate<i64>(scale),
    });

    train_mnist<i64, scale>(model);
}

void lenet_float() {
    auto model = Sequential<float>({
        new Conv2D<float, 0>(1, 6, 5, 1, 1),
        new ReLU<float>(),
        new MaxPool2D<float>(2),
        new Conv2D<float, 0>(6, 16, 5, 1),
        new ReLU<float>(),
        new MaxPool2D<float>(2),
        new Conv2D<float, 0>(16, 120, 5),
        new ReLU<float>(),
        new Flatten<float>(),
        new FC<float, 0>(120, 84),
        new ReLU<float>(),
        new FC<float, 0>(84, 10),
    });

    train_mnist<float, 0>(model);
}

void threelayer_int() {
    const u64 scale = 24;
    auto model = Sequential<i64>({
        new Conv2D<i64, scale>(3, 64, 5, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        new Conv2D<i64, scale>(64, 64, 5, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        new Conv2D<i64, scale>(64, 64, 5, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        new Flatten<i64>(),
        new FC<i64, scale>(64, 10),
        new Truncate<i64>(scale),
    });

    train_cifar10<i64, scale>(model);
}

void threelayer_float() {
    auto model = Sequential<double>({
        new Conv2D<double, 0>(3, 64, 5, 1),
        new BatchNorm2d<double, 0>(64),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Conv2D<double, 0>(64, 64, 5, 1),
        new BatchNorm2d<double, 0>(64),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Conv2D<double, 0>(64, 64, 5, 1),
        new BatchNorm2d<double, 0>(64),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Flatten<double>(),
        new FC<double, 0>(64, 10),
    });
    train_cifar10<double, 0>(model);
}

void piranha_vgg_int()
{
    const u64 scale = 24;
    auto model = Sequential<i64>({
        new Conv2D<i64, scale>(3, 64, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(64, 64, 3, 1),
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        new Conv2D<i64, scale>(64, 128, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(128, 128, 3, 1),
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        new Conv2D<i64, scale>(128, 256, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(256, 256, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(256, 256, 3, 1),
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        new Conv2D<i64, scale>(256, 512, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(512, 512, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(512, 512, 3, 1),
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        new Conv2D<i64, scale>(512, 512, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(512, 512, 3, 1),
        new ReLUTruncate<i64>(scale),
        new Conv2D<i64, scale>(512, 512, 3, 1),
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        new Flatten<i64>(),
        new FC<i64, scale>(512, 256),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(256, 256),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(256, 10),
        new Truncate<i64>(scale),
    });

    train_cifar10<i64, scale>(model);
}
