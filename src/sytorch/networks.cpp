#include <sytorch/layers/layers.h>
#include <sytorch/train.h>
#include <filesystem>
#include <sytorch/sequential.h>

uint8_t *readFile(std::string filename, size_t* input_size)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(filename, std::ios::binary);
    // const int fileDesc = open(filename.c_str(), O_RDONLY/*| O_DIRECT*/);
    size_t size_in_bytes = std::filesystem::file_size(filename);
    *input_size = size_in_bytes;
    uint8_t *mem_bytes = new uint8_t[size_in_bytes];

    file.read((char*) mem_bytes, size_in_bytes);
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    // std::cout << "Time to read file in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    // std::cout << "File size: " << size_in_bytes << std::endl;
    return mem_bytes;
}

// taken from Neha's code to load model
// different from the similar function in main.cpp as this function assumes that the file contains integers with some scale
// function inside main.cpp assumes that the file contains float32 numbers. Scale can be appropriately chosen by the program
template <typename T>void readWeights(Sequential<T> &model, string weightsFile) {
    size_t weightsSize;
    auto W = (GroupElement*) readFile(weightsFile, &weightsSize);
    auto N = weightsSize / sizeof(GroupElement);
    int wIdx = 0;
    for(int i = 0; i < model.layers.size(); i++) {
        if(model.layers[i]->name.find("Conv2D") != std::string::npos || model.layers[i]->name.find("FC") != std::string::npos) {
            auto& weights = model.layers[i]->getweights();
            weights.data = (int64_t*) &W[wIdx];
            auto wSize = weights.d1 * weights.d2;
            wIdx += wSize;
            auto& bias = model.layers[i]->getbias();
            bias.data = (int64_t*) &W[wIdx];
            wSize = bias.size;
            wIdx += wSize;
        }
    }
    always_assert(wIdx == N);
}

void lenet_int() {
    const u64 scale = 24;
    auto model = Sequential<i64>({
        new Conv2D<i64>(1, 8, 5),
        new ReLU<i64>(),
        new MaxPool2D<i64>(2),
        new Conv2D<i64>(8, 16, 5),
        new ReLU<i64>(),
        new MaxPool2D<i64>(2),
        new Flatten<i64>(),
        new FC<i64>(256, 128),
        new ReLU<i64>(),
        new FC<i64>(128, 10),
    });

    // readWeights(model, "initial_weights_lenet.dat");
    train_mnist<i64, scale>(model);
}


void lenet_pirhana_int() {
    const u64 scale = 24;
    auto model = Sequential<i64>({
        new Conv2D<i64>(1, 20, 5),
        new ReLU<i64>(),
        new MaxPool2D<i64>(2),
        new Conv2D<i64>(20, 50, 5),
        new ReLU<i64>(),
        new MaxPool2D<i64>(2),
        new Flatten<i64>(),
        new FC<i64>(800, 500),
        new ReLU<i64>(),
        new FC<i64>(500, 10),
    });

    train_mnist<i64, scale>(model);
}

void lenet_float() {
    auto model = Sequential<float>({
        new Conv2D<float>(1, 6, 5, 1, 1),
        new ReLU<float>(),
        new MaxPool2D<float>(2),
        new Conv2D<float>(6, 16, 5, 1),
        new ReLU<float>(),
        new MaxPool2D<float>(2),
        new Conv2D<float>(16, 120, 5),
        new ReLU<float>(),
        new Flatten<float>(),
        new FC<float>(120, 84),
        new ReLU<float>(),
        new FC<float>(84, 10),
    });

    train_mnist<float, 0>(model);
}

void threelayer_int() {
    const u64 scale = 24;
    auto model = Sequential<i64>({
        new Conv2D<i64>(3, 64, 5, 1),
        new ReLU<i64>(),
        new MaxPool2D<i64>(3, 0, 2),
        new Conv2D<i64>(64, 64, 5, 1),
        new ReLU<i64>(),
        new MaxPool2D<i64>(3, 0, 2),
        new Conv2D<i64>(64, 64, 5, 1),
        new ReLU<i64>(),
        new MaxPool2D<i64>(3, 0, 2),
        new Flatten<i64>(),
        new FC<i64>(64, 10),
    });

    // readWeights(model, "initial_weights_3layer.dat");
    train_cifar10<i64, scale>(model);
}

void threelayer_float() {
    auto model = Sequential<double>({
        new Conv2D<double>(3, 64, 5, 1),
        new BatchNorm2d<double>(64),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Conv2D<double>(64, 64, 5, 1),
        new BatchNorm2d<double>(64),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Conv2D<double>(64, 64, 5, 1),
        new BatchNorm2d<double>(64),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Flatten<double>(),
        new FC<double>(64, 10),
    });
    train_cifar10<double, 0>(model);
}

void alexnet_float() {
    auto model = Sequential<float>({
        new Conv2D<float>(3, 96, 11, 9, 4),
        new MaxPool2D<float>(3, 0, 2),
        new ReLU<float>(),
        new Conv2D<float>(96, 256, 5, 1, 1),
        new ReLU<float>(),
        new MaxPool2D<float>(2, 0, 1),
        new Conv2D<float>(256, 384, 3, 1, 1),
        new ReLU<float>(),
        new Conv2D<float>(384, 384, 3, 1, 1),
        new ReLU<float>(),
        new Conv2D<float>(384, 256, 3, 1, 1),
        new ReLU<float>(),
        new Flatten<float>(),
        new FC<float>(256, 256),
        new ReLU<float>(),
        new FC<float>(256, 256),
        new ReLU<float>(),
        new FC<float>(256, 10),
    });

    train_cifar10<float, 0>(model);
}

void secureml_int()
{
    auto model = Sequential<i64>({
        new Flatten<i64>(),
        new FC<i64>(784, 128),
        new ReLU<i64>(),
        new FC<i64>(128, 128),
        new ReLU<i64>(),
        new FC<i64>(128, 10),
    });

    train_mnist<i64, 16>(model);
}

// void piranha_vgg_int()
// {
//     const u64 scale = 24;
//     auto model = Sequential<i64>({
//         new Conv2D<i64>(3, 64, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(64, 64, 3, 1),
//         new SumPool2D<i64>(2, 0, 2),
//         new ReLUTruncate<i64>(scale+2),
//         new Conv2D<i64>(64, 128, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(128, 128, 3, 1),
//         new SumPool2D<i64>(2, 0, 2),
//         new ReLUTruncate<i64>(scale+2),
//         new Conv2D<i64>(128, 256, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(256, 256, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(256, 256, 3, 1),
//         new SumPool2D<i64>(2, 0, 2),
//         new ReLUTruncate<i64>(scale+2),
//         new Conv2D<i64>(256, 512, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(512, 512, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(512, 512, 3, 1),
//         new SumPool2D<i64>(2, 0, 2),
//         new ReLUTruncate<i64>(scale+2),
//         new Conv2D<i64>(512, 512, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(512, 512, 3, 1),
//         new ReLU<i64>(),
//         new Conv2D<i64>(512, 512, 3, 1),
//         new SumPool2D<i64>(2, 0, 2),
//         new ReLUTruncate<i64>(scale+2),
//         new Flatten<i64>(),
//         new FC<i64>(512, 256),
//         new ReLU<i64>(),
//         new FC<i64>(256, 256),
//         new ReLU<i64>(),
//         new FC<i64>(256, 10),
//         new Truncate<i64>(scale),
//     });

//     train_cifar10<i64>(model);
// }

// void piranha_vgg_float()
// {
//     const u64 scale = 0;
//     using T = double;
//     auto model = Sequential<T>({
//         new Conv2D<T>(3, 64, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(64, 64, 3, 1),
//         new SumPool2D<T>(2, 0, 2),
//         new ReLUTruncate<T>(scale+2),
//         new Conv2D<T>(64, 128, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(128, 128, 3, 1),
//         new SumPool2D<T>(2, 0, 2),
//         new ReLUTruncate<T>(scale+2),
//         new Conv2D<T>(128, 256, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(256, 256, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(256, 256, 3, 1),
//         new SumPool2D<T>(2, 0, 2),
//         new ReLUTruncate<T>(scale+2),
//         new Conv2D<T>(256, 512, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(512, 512, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(512, 512, 3, 1),
//         new SumPool2D<T>(2, 0, 2),
//         new ReLUTruncate<T>(scale+2),
//         new Conv2D<T>(512, 512, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(512, 512, 3, 1),
//         new ReLUTruncate<T>(scale),
//         new Conv2D<T>(512, 512, 3, 1),
//         new SumPool2D<T>(2, 0, 2),
//         new ReLUTruncate<T>(scale+2),
//         new Flatten<T>(),
//         new FC<T>(512, 256),
//         new ReLUTruncate<T>(scale),
//         new FC<T>(256, 256),
//         new ReLUTruncate<T>(scale),
//         new FC<T>(256, 10),
//         new Truncate<T>(scale),
//     });

//     train_cifar10<T>(model);
// }
