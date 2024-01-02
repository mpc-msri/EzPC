#include <iostream>
#include <vector>
#include "layers.h"
#include "softmax.h"
#include "mnist.h"
#include <cmath>
#include <iomanip>

const u64 scale = 12;

void test_float() {
    auto model = Sequential<double>({
        new Flatten<double>(),
        new FC<double, 0>(784, 500),
        new ReLU<double>(),
        new FC<double, 0>(500, 10),
        new ReLU<double>(),
    });

    // ((FC<double, 0> *)(model.layers[3]))->weight.print();

    Tensor4D<double> image(1, 28, 28, 1);
    for(int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            image(0, i, j, 0) = 1.0;//train_image[0][i * 28 + j];
        }
    }

    model.forward(image);
    model.activation.print();
    std::cout << "== Forward Pass complete ==" << std::endl;
    Tensor4D<double> e(1, 10, 1, 1);
    double max = model.activation(0, 0, 0, 0);
    for(u64 j = 1; j < 10; ++j) {
        if(model.activation(0, j, 0, 0) > max) {
            max = model.activation(0, j, 0, 0);
        }
    }
    double den = 0.0;
    for(u64 j = 0; j < 10; ++j) {
        e(0, j, 0, 0) = model.activation(0, j, 0, 0) - max;
        den += exp(e(0, j, 0, 0));
    }

    for(u64 j = 0; j < 10; ++j) {
        double x = e(0, j, 0, 0);
        e(0, j, 0, 0) = exp(x) / den;
    }
    e(0, 0, 0, 0) -= 1.0;
    std::cout << "e: ";
    e.print();
    model.backward(e);
    // ((FC<double, 0> *)(model.layers[1]))->weightGrad.print();
    // ((FC<double, 0> *)(model.layers[1]))->weight.print();
    model.forward(image);
    model.activation.print();
}

void test_int() {
    auto model = Sequential<i64>({
        new Flatten<i64>(),
        new FC<i64, scale>(784, 500),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(500, 10),
        new ReLUTruncate<i64>(scale)
    });

    Tensor4D<i64> image(1, 28, 28, 1);
    for(int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            image(0, i, j, 0) = (1ULL << scale);
        }
    }

    model.forward(image);
    std::cout << "== Forward Pass complete ==" << std::endl;
    Tensor4D<i64> e(1, 10, 1, 1);
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
    e(0, 0, 0, 0) -= (1ULL << scale);
    std::cout << "e: ";
    e.print();
    model.backward(e);
}

void vgg16_piranha_keysize_llama() {
    LlamaKey<i64>::serverkeysize = 0;
    LlamaKey<i64>::clientkeysize = 0;
    auto model = Sequential<i64>({
        new Conv2D<i64, scale, LlamaKey<i64>>(3, 64, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 3, 1),
        new AvgPool2D<i64, scale, LlamaKey<i64>>(2, 0, 2),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 128, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(128, 128, 3, 1),
        new AvgPool2D<i64, scale, LlamaKey<i64>>(2, 0, 2),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(128, 256, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(256, 256, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(256, 256, 3, 1),
        new AvgPool2D<i64, scale, LlamaKey<i64>>(2, 0, 2),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(256, 512, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(512, 512, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(512, 512, 3, 1),
        new AvgPool2D<i64, scale, LlamaKey<i64>>(2, 0, 2),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(512, 512, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(512, 512, 3, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Conv2D<i64, scale, LlamaKey<i64>>(512, 512, 3, 1),
        new AvgPool2D<i64, scale, LlamaKey<i64>>(2, 0, 2),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new Flatten<i64>(),
        new FC<i64, scale, false, LlamaKey<i64>>(512, 256),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new FC<i64, scale, false, LlamaKey<i64>>(256, 256),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new FC<i64, scale, false, LlamaKey<i64>>(256, 10),
        new Truncate<i64, LlamaKey<i64>>(scale),
    });

    Tensor4D<i64> trainImage(128, 32, 32, 3);
    Tensor4D<i64> e(128, 10, 1, 1);

    model.forward(trainImage);
    std::cout << ">>>>>>> FORWARD DONE <<<<<<<<" << std::endl;
    model.backward(e);

    double gb = 1024ULL * 1024 * 1024 * 8ULL;
    std::cout << "Server key size: " << LlamaKey<i64>::serverkeysize / gb << std::endl;
    std::cout << "Client key size: " << LlamaKey<i64>::clientkeysize / gb << std::endl;

}

void test_conv_float()
{
    std::cout << "=== Running Floating Point CNN Training ===" << std::endl;
    auto conv1 = new Conv2D<double, 0>(1, 1, 2, 1, 1);
    auto avgpool1 = new AvgPool2D<double, 0>(2, 0, 2);

    auto model = Sequential<double>({
        conv1,
        avgpool1,
    });

    conv1->filter(0, 0) = 1;
    conv1->filter(0, 1) = 2;
    conv1->filter(0, 2) = 3;
    conv1->filter(0, 3) = 4;
    conv1->bias(0) = 0;

    // std::cout << "filter: ";
    // conv1->filter.print();
    Tensor4D<double> a(1, 3, 3, 1);
    a.fill(1);

    model.forward(a);

    // std::cout << "act: ";
    // model.activation.print();

    Tensor4D<double> e(1, 2, 2, 1);
    e(0, 0, 0, 0) = 1;
    e(0, 0, 1, 0) = 1;
    e(0, 1, 0, 0) = 1;
    e(0, 1, 1, 0) = 1;
    // // model.activation.print();
    model.backward(e);
    model.forward(a);
    model.backward(e);
    std::cout << "filter grad: ";
    conv1->filterGrad.print();
    std::cout << "filter: ";
    conv1->filter.print();
    std::cout << "input grad: ";
    conv1->inputDerivative.print();
}

void lenet_float() {

    srand(time(NULL));
    std::cout << "=== Running Floating Point Training ===" << std::endl;

    auto model = Sequential<double>({
        new Conv2D<double, 0>(1, 6, 5, 1, 1),
        new ReLU<double>(),
        new AvgPool2D<double, 0>(2),
        new Conv2D<double, 0>(6, 16, 5, 1),
        new ReLU<double>(),
        new AvgPool2D<double, 0>(2),
        new Conv2D<double, 0>(16, 120, 5),
        new ReLU<double>(),
        new Flatten<double>(),
        new FC<double, 0>(120, 84),
        new ReLU<double>(),
        new FC<double, 0>(84, 10),
    });

    Tensor4D<double> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k];
            }
        }
    }

    Tensor4D<double> e(batchSize, 10, 1, 1);
    Tensor4D<double> trainImage(batchSize, 28, 28, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        for(u64 i = 0; i < trainLen; i += batchSize) {
            // fetch image
            for(int b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 28; ++j) {
                    for(u64 k = 0; k < 28; ++k) {
                        trainImage(b, j, k, 0) = train_image[i+b][j * 28 + k];
                    }
                }
            }
            model.forward(trainImage);
            softmax<double, 0>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, train_label[i+b], 0, 0) -= (1.0/batchSize);
            }
            model.backward(e);
        }
        model.forward(testSet);
        u64 correct = 0;
        for(u64 i = 0; i < 10000; i++) {
            if (model.activation.argmax(i) == test_label[i]) {
                correct++;
            }
        }
        std::cout << "Epoch: " << epoch << " Accuracy: " << correct/100.0 << std::endl;
    }
}

void lenet_int() {

    srand(time(NULL));
    std::cout << "=== Running Fixed-Point Training ===" << std::endl;

    auto model = Sequential<i64>({
        new Conv2D<i64, scale>(1, 6, 5, 1, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(2),
        new Conv2D<i64, scale>(6, 16, 5, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(2),
        new Conv2D<i64, scale>(16, 120, 5),
        new ReLUTruncate<i64>(scale),
        new Flatten<i64>(),
        new FC<i64, scale>(120, 84),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(84, 10),
        new Truncate<i64>(scale),
    });

    Tensor4D<i64> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    Tensor4D<i64> e(batchSize, 10, 1, 1);
    Tensor4D<i64> trainImage(batchSize, 28, 28, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        for(u64 i = 0; i < trainLen; i += batchSize) {
            // fetch image
            for(int b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 28; ++j) {
                    for(u64 k = 0; k < 28; ++k) {
                        trainImage(b, j, k, 0) = train_image[i+b][j * 28 + k] * (1ULL << scale);
                    }
                }
            }
            model.forward(trainImage);
            softmax<i64, scale>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, train_label[i+b], 0, 0) -= ((1ULL<<scale)/batchSize);
            }
            model.backward(e);
        }
        model.forward(testSet);
        u64 correct = 0;
        for(u64 i = 0; i < 10000; i++) {
            if (model.activation.argmax(i) == test_label[i]) {
                correct++;
            }
        }
        std::cout << "Epoch: " << epoch << " Accuracy: " << correct/100.0 << std::endl;
    }
}

Tensor4D<double> diff(Tensor4D<i64> &t, u64 scale, Tensor4D<double> &t2) {
    Tensor4D<double> ret(t.d1, t.d2, t.d3, t.d4);
    for(u64 i = 0; i < t.d1; ++i) {
        for(u64 j = 0; j < t.d2; ++j) {
            for(u64 k = 0; k < t.d3; ++k) {
                for(u64 l = 0; l < t.d4; ++l) {
                    ret(i, j, k, l) = t2(i, j, k, l) - (t(i, j, k, l) / ((double)(1ULL<<scale)));
                }
            }
        }
    }
    return ret;
}

Tensor2D<double> diff(Tensor2D<i64> &t, u64 scale, Tensor2D<double> &t2) {
    Tensor2D<double> ret(t.d1, t.d2);
    for(u64 i = 0; i < t.d1; ++i) {
        for(u64 j = 0; j < t.d2; ++j) {
            ret(i, j) = t2(i, j) - (t(i, j) / ((double)(1ULL<<scale)));
        }
    }
    return ret;
}

Tensor<double> diff(Tensor<i64> &t, u64 scale, Tensor<double> &t2) {
    Tensor<double> ret(t.size);
    for(u64 i = 0; i < t.size; ++i) {
        ret(i) = t2(i) - (t(i) / ((double)(1ULL<<scale)));
    }
    return ret;
}

void single_layer_test() {
    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();
    const u64 batchSize = 2;
    const u64 scale = 12;

    auto fc1 = new FC<double, 0>(3072, 10);
    auto rt  = new ReLU<double>();
    auto model = Sequential<double>({
        new Flatten<double>(),
        fc1,
        rt,
    });

    fc1->weight.fill(0.009021);
    fc1->bias.fill(0.009021);

    auto fc1_fp = new FC<i64, scale>(3072, 10);
    auto rt_fp  = new ReLUTruncate<i64>(scale);
    auto model_fp = Sequential<i64>({
        new Flatten<i64>(),
        fc1_fp,
        rt_fp,
    });

    fc1_fp->weight.fill(0.009021 * (1ULL<<scale));
    fc1_fp->bias.fill(0.009021 * (1ULL<<(2*scale)));

    Tensor4D<double> trainImage(batchSize, 32, 32, 3);
    Tensor4D<i64> trainImage_fp(batchSize, 32, 32, 3);
    Tensor4D<double> e(batchSize, 10, 1, 1);
    Tensor4D<i64> e_fp(batchSize, 10, 1, 1);
    Tensor<u64> dist(10);
    dist.fill(0);
    for(u64 i = 0; i < trainLen; i += batchSize) {
        for(u64 b = 0; b < batchSize; ++b) {
            for(u64 j = 0; j < 32; ++j) {
                for(u64 k = 0; k < 32; ++k) {
                    for(u64 l = 0; l < 3; ++l) {
                        trainImage(b, j, k, l) = dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0;
                        auto pixel = (dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1ULL<<(scale));
                        trainImage_fp(b, j, k, l) = pixel + rand_float();
                    }
                }
            }
        }

        model.forward(trainImage);
        model_fp.forward(trainImage_fp);
        softmax<double, 0>(model.activation, e);
        softmax<i64, scale>(model_fp.activation, e_fp);
        for(u64 b = 0; b < batchSize; ++b) {
            e_fp(b, dataset.training_labels[i+b], 0, 0) -= (1ULL<<(scale))/batchSize;
            e(b, dataset.training_labels[i+b], 0, 0) -= (1.0/batchSize);
            dist(dataset.training_labels[i+b])++;
            // for(u64 j = 0; j < 10; ++j) {
            //     e(b, j, 0, 0) = e_fp(b, j, 0, 0) / ((double)(1ULL<<scale));
            // }
        }
        model.backward(e);
        model_fp.backward(e_fp);
        // at i = 24 * batchSize, something special happens in the first element of bias - sign of the relu is flipped in fp vs float causing huge difference
        if (i == 3 * batchSize) {
            // auto d = diff(e_fp, scale, e);
            // auto d = diff(fc1_fp->weight, scale, fc1->weight);
            std::cout << "Stopping at i = " << i << std::endl;
            std::cout << "FP\nbias = ";
            fc1_fp->bias.print(2*scale);
            std::cout << "e = \n";
            // e_fp.print(scale);
            rt_fp->inputDerivative.print(scale);
            std::cout << "fc act =\n";
            fc1_fp->activation.print(2*scale);
            std::cout << "Float\nbias = ";
            fc1->bias.print();
            std::cout << "e = \n";
            // e.print();
            rt->inputDerivative.print();
            std::cout << "fc act =\n";
            fc1->activation.print();
            auto d = diff(fc1_fp->bias, 2*scale, fc1->bias);
            std::cout << "Diff\n";
            d.print();
            // std::cout << 1.0/(1ULL<<scale) << std::endl;
            dist.print();
            break;
        }
    }
}


void single_layer_test_norelu() {
    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();
    const u64 batchSize = 2;
    const u64 scale = 12;

    auto fc1 = new FC<double, 0>(3072, 10);
    auto model = Sequential<double>({
        new Flatten<double>(),
        fc1,
    });

    fc1->weight.fill(0.009021);
    fc1->bias.fill(0.009021);

    auto fc1_fp = new FC<i64, scale>(3072, 10);
    auto rt_fp  = new Truncate<i64>(scale);
    auto model_fp = Sequential<i64>({
        new Flatten<i64>(),
        fc1_fp,
        rt_fp,
    });

    for(u64 i = 0; i < 3072; ++i) {
        for(u64 j = 0; j < 10; ++j) {
            fc1_fp->weight(i, j) = (0.009021 * (1ULL<<scale)) + rand_float();
        }
    }

    for(u64 i = 0; i < 10; ++i) {
        fc1_fp->bias(i) = (0.009021 * (1ULL<<(2*scale))) + rand_float();
    }

    Tensor4D<double> trainImage(batchSize, 32, 32, 3);
    Tensor4D<i64> trainImage_fp(batchSize, 32, 32, 3);
    Tensor4D<double> e(batchSize, 10, 1, 1);
    Tensor4D<i64> e_fp(batchSize, 10, 1, 1);
    Tensor<u64> dist(10);
    dist.fill(0);
    for(u64 i = 0; i < trainLen; i += batchSize) {
        for(u64 b = 0; b < batchSize; ++b) {
            for(u64 j = 0; j < 32; ++j) {
                for(u64 k = 0; k < 32; ++k) {
                    for(u64 l = 0; l < 3; ++l) {
                        trainImage(b, j, k, l) = dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0;
                        auto pixel = (dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1ULL<<(scale));
                        trainImage_fp(b, j, k, l) = pixel + rand_float();
                    }
                }
            }
        }

        model.forward(trainImage);
        model_fp.forward(trainImage_fp);
        softmax<double, 0>(model.activation, e);
        softmax<i64, scale>(model_fp.activation, e_fp);
        for(u64 b = 0; b < batchSize; ++b) {
            e_fp(b, dataset.training_labels[i+b], 0, 0) -= (1ULL<<(scale))/batchSize;
            e(b, dataset.training_labels[i+b], 0, 0) -= (1.0/batchSize);
            dist(dataset.training_labels[i+b])++;
        }
        model.backward(e);
        model_fp.backward(e_fp);
        // at i = 24 * batchSize, something special happens in the first element of bias - sign of the relu is flipped in fp vs float causing huge difference
        if (i == 30000 * batchSize) {
            // auto d = diff(e_fp, scale, e);
            // auto d = diff(fc1_fp->weight, scale, fc1->weight);
            std::cout << "Stopping at i = " << i << std::endl;
            std::cout << "FP\n";
            // std::cout << "bias = \n";
            // fc1_fp->bias.print(2*scale);
            // std::cout << "e = \n";
            // e_fp.print(scale);
            // std::cout << "act =\n";
            // model_fp.activation.print(scale);
            std::cout << "weight = \n";
            std::cout << fc1_fp->weight(0, 0) / ((double)(1ULL<<scale)) << " ";
            // std::cout << fc1_fp->weight(0, 0) << " ";
            std::cout << fc1_fp->weight(0, 1) / ((double)(1ULL<<scale)) << " ";
            std::cout << fc1_fp->weight(0, 2) / ((double)(1ULL<<scale)) << " ";
            std::cout << fc1_fp->weight(0, 3) / ((double)(1ULL<<scale)) << std::endl;
            // std::cout << "weightGrad = \n";
            // std::cout << fc1_fp->weightGrad(0, 0) / ((double)(1ULL<<(2*scale))) << " ";
            // std::cout << fc1_fp->weightGrad(0, 1) / ((double)(1ULL<<(2*scale))) << " ";
            // std::cout << fc1_fp->weightGrad(0, 2) / ((double)(1ULL<<(2*scale))) << " ";
            // std::cout << fc1_fp->weightGrad(0, 3) / ((double)(1ULL<<(2*scale))) << std::endl;
            // std::cout << "Vw = \n";
            // std::cout << fc1_fp->Vw(0, 0) / ((double)(1ULL<<(2*scale))) << " ";
            // std::cout << fc1_fp->Vw(0, 1) / ((double)(1ULL<<(2*scale))) << " ";
            // std::cout << fc1_fp->Vw(0, 2) / ((double)(1ULL<<(2*scale))) << " ";
            // std::cout << fc1_fp->Vw(0, 3) / ((double)(1ULL<<(2*scale))) << std::endl;
            std::cout << "Float\n";
            // std::cout << "bias = \n";
            // fc1->bias.print();
            // std::cout << "e = \n";
            // e.print();
            // std::cout << "act =\n";
            // model.activation.print();
            std::cout << "weight = \n";
            std::cout << fc1->weight(0, 0) << " ";
            std::cout << fc1->weight(0, 1) << " ";
            std::cout << fc1->weight(0, 2) << " ";
            std::cout << fc1->weight(0, 3) << std::endl;
            // std::cout << "weightGrad = \n";
            // std::cout << fc1->weightGrad(0, 0) << " ";
            // std::cout << fc1->weightGrad(0, 1) << " ";
            // std::cout << fc1->weightGrad(0, 2) << " ";
            // std::cout << fc1->weightGrad(0, 3) << std::endl;
            // std::cout << "Vw = \n";
            // std::cout << fc1->Vw(0, 0) << " ";
            // std::cout << fc1->Vw(0, 1) << " ";
            // std::cout << fc1->Vw(0, 2) << " ";
            // std::cout << fc1->Vw(0, 3) << std::endl;
            break;
        }
    }

}


void threelayer_keysize_llama() {

    const u64 scale = 24;
    LlamaKey<i64>::verbose = true;
    LlamaKey<i64>::probablistic = true;
    LlamaKey<i64>::bw = 64;
    const u64 minibatch = 100;

    auto model = Sequential<i64>({
        new Conv2D<i64, scale, LlamaKey<i64>>(3, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
        new Flatten<i64>(),
        new FC<i64, scale, LlamaKey<i64>>(64, 10),
        new Truncate<i64, LlamaKey<i64>>(scale),
    });

    double gb = 1024ULL * 1024 * 1024 * 8ULL;

    Tensor4D<i64> trainImage(minibatch, 32, 32, 3);
    LlamaKey<i64>::serverkeysize = 0;
    LlamaKey<i64>::clientkeysize = 0;
    std::cout << ">>>>>>> FORWARD START<<<<<<<<" << std::endl;
    model.forward(trainImage);
    std::cout << ">>>>>>> FORWARD DONE <<<<<<<<" << std::endl;
    std::cout << "Server key size: " << LlamaKey<i64>::serverkeysize / gb << " GB" << std::endl;
    std::cout << "Client key size: " << LlamaKey<i64>::clientkeysize / gb << " GB" << std::endl;
    Tensor4D<i64> e(minibatch, model.activation.d2, model.activation.d3, model.activation.d4);
    LlamaKey<i64>::serverkeysize = 0;
    LlamaKey<i64>::clientkeysize = 0;
    std::cout << ">>>>>>> BACKWARD START <<<<<<<<" << std::endl;
    model.backward(e);
    std::cout << ">>>>>>> BACKWARD DONE <<<<<<<<" << std::endl;
    std::cout << "Server key size: " << LlamaKey<i64>::serverkeysize / gb << " GB" << std::endl;
    std::cout << "Client key size: " << LlamaKey<i64>::clientkeysize / gb << " GB" << std::endl;
}


void llama_test(int party) {
    srand(time(NULL));
    const u64 scale = 16;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto conv1 = new Conv2D<u64, scale, Llama<u64>>(3, 64, 5, 1);
    auto relu1 = new ReLUTruncate<u64, Llama<u64>>(scale);
    auto model = Sequential<u64>({
        conv1,
        relu1,
    });

    Tensor4D<u64> trainImage(2, 32, 32, 3); // 1 images with server and 1 with client

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    Llama<u64>::output(model.activation);
    Llama<u64>::finalize();
    if (LlamaConfig::party != 1)
        model.activation.print();
}

void llama_relu2round_test(int party) {
    srand(time(NULL));
    const u64 scale = 16;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto relu1 = new ReLU<u64, Llama<u64>>();
    auto model = Sequential<u64>({
        relu1,
    });

    Tensor4D<u64> trainImage(2, 2, 2, 1); // 1 images with server and 1 with client
    trainImage(0, 0, 0, 0) = 5;
    trainImage(0, 0, 1, 0) = 7;
    trainImage(0, 1, 0, 0) = 12;
    trainImage(0, 1, 1, 0) = 15;
    trainImage(1, 0, 0, 0) = -5;
    trainImage(1, 0, 1, 0) = -7;
    trainImage(1, 1, 0, 0) = -12;
    trainImage(1, 1, 1, 0) = -15;

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    Llama<u64>::output(model.activation);
    Llama<u64>::output(relu1->drelu);
    Llama<u64>::finalize();
    if (LlamaConfig::party != 1) {
        model.activation.print();
        relu1->drelu.print();
    }
}

void llama_relu_old_test(int party) {
    srand(time(NULL));
    const u64 scale = 16;
    LlamaConfig::party = party;
    LlamaExtended<u64>::init();
    auto relu1 = new ReLU<u64, LlamaExtended<u64>>();
    auto model = Sequential<u64>({
        relu1,
    });

    Tensor4D<u64> trainImage(2, 2, 2, 1); // 1 images with server and 1 with client
    trainImage(0, 0, 0, 0) = 5;
    trainImage(0, 0, 1, 0) = 7;
    trainImage(0, 1, 0, 0) = 12;
    trainImage(0, 1, 1, 0) = 15;
    trainImage(1, 0, 0, 0) = -5;
    trainImage(1, 0, 1, 0) = -7;
    trainImage(1, 1, 0, 0) = -12;
    trainImage(1, 1, 1, 0) = -15;

    LlamaExtended<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    LlamaExtended<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    LlamaExtended<u64>::output(model.activation);
    LlamaExtended<u64>::output(relu1->drelu);
    LlamaExtended<u64>::finalize();
    if (LlamaConfig::party != 1) {
        model.activation.print();
        relu1->drelu.print();
    }
}

void llama_test_vgg(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto model = Sequential<u64>({
        new Conv2D<u64, scale, Llama<u64>>(3, 64, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(64, 64, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(64, 128, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(128, 128, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(128, 256, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(256, 256, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(256, 256, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(256, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Flatten<u64>(),
        new FC<u64, scale, Llama<u64>>(512, 256),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new FC<u64, scale, Llama<u64>>(256, 256),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new FC<u64, scale, Llama<u64>>(256, 10),
        new Truncate<u64, Llama<u64>>(scale),
    });

    // Tensor4D<u64> trainImage(2, 1, 2, 1); // 1 images with server and 1 with client
    Tensor4D<u64> trainImage(2, 32, 32, 3); // 1 images with server and 1 with client
    Tensor4D<u64> e(2, 10, 1, 1); // 1 images with server and 1 with client

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    pirhana_softmax(model.activation, e, scale);
    EndComputation();
    Llama<u64>::output(e);
    Llama<u64>::finalize();
    if (LlamaConfig::party != 1)
        e.print();
}


void llama_test_small(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto rt = new ReLUTruncate<u64, Llama<u64>>(scale);
    auto fc = new Conv2D<u64, scale, Llama<u64>>(1, 1, 2);
    auto model = Sequential<u64>({
        fc,
        rt,
        new Flatten<u64>(),
    });

    auto fc_ct = new Conv2D<i64, scale>(1, 1, 2);
    fc_ct->filter.copy(fc->filter);
    fc_ct->bias.copy(fc->bias);
    auto rt_ct = new ReLUTruncate<i64>(scale);

    auto model_ct = Sequential<i64>({
        fc_ct,
        rt_ct,
        new Flatten<i64>(),
    });

    // Tensor4D<u64> trainImage(2, 1, 2, 1); // 1 images with server and 1 with client
    Tensor4D<u64> trainImage(2, 4, 4, 1); // 1 images with server and 1 with client
    trainImage.fill((1ULL<<(scale+1)));
    trainImage(0, 0, 0, 0) = (1ULL<<(scale+2));
    trainImage(0, 1, 2, 0) = (1ULL<<(scale+2));
    trainImage(0, 2, 3, 0) = (1ULL<<(scale+2));
    trainImage(0, 3, 1, 0) = (1ULL<<(scale+2));
    Tensor4D<i64> trainImage_ct(2, 4, 4, 1);
    trainImage_ct.copy(trainImage);
    Tensor4D<u64> e(2, 9, 1, 1); // 1 images with server and 1 with client
    Tensor4D<i64> e_ct(2, 9, 1, 1);

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    pirhana_softmax(model.activation, e, scale);
    model.backward(e);
    EndComputation();
    // Llama<u64>::output(rt->drelu);
    Llama<u64>::output(model.activation);
    Llama<u64>::output(e);
    Llama<u64>::output(fc->filter);
    Llama<u64>::output(fc->bias);
    if (LlamaConfig::party != 1) {
        // rt->drelu.print();
        std::cout << "Secure Computation Output = \n";
        model.activation.print<i64>();
        e.print<i64>(); // eprint hehe
        fc->filter.print<i64>();
        fc->bias.print<i64>();
    }
    Llama<u64>::finalize();

    // comparison with ct
    model_ct.forward(trainImage_ct);
    pirhana_softmax_ct(model_ct.activation, e_ct, scale);
    model_ct.backward(e_ct);
    if (LlamaConfig::party == 1) {
        std::cout << "Plaintext Computation Output = \n";
        model_ct.activation.print();
        e_ct.print();
        fc_ct->filter.print();
        fc_ct->bias.print();
    }
}


void cifar10_float_test() {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();
    const u64 batchSize = 100;
    const u64 scale = 12;

    auto conv1 = new Conv2D<double, 0>(3, 64, 5, 1);
    auto conv2 = new Conv2D<double, 0>(64, 64, 5, 1);
    auto conv3 = new Conv2D<double, 0>(64, 64, 5, 1);
    auto fc1 = new FC<double, 0>(64, 10);
    auto rt3 = new ReLU<double>();
    auto model = Sequential<double>({
        /// 3 Layer from Gupta et al
        conv1,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        conv2,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        conv3,
        rt3,
        new MaxPool2D<double>(3, 0, 2),
        new Flatten<double>(),
        fc1,
    });

    auto conv1_fp = new Conv2D<i64, scale>(3, 64, 5, 1);
    auto conv2_fp = new Conv2D<i64, scale>(64, 64, 5, 1);
    auto conv3_fp = new Conv2D<i64, scale>(64, 64, 5, 1);
    auto fc1_fp = new FC<i64, scale>(64, 10);
    auto rt3_fp = new ReLUTruncate<i64>(scale);
    auto t_last = new Truncate<i64>(scale);
    auto model_fp = Sequential<i64>({
        /// 3 Layer from Gupta et al
        conv1_fp,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        conv2_fp,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        conv3_fp,
        rt3_fp,
        new MaxPool2D<i64>(3, 0, 2),
        new Flatten<i64>(),
        fc1_fp,
        t_last,
    });

    for(int i = 0; i < conv1_fp->filter.d1; ++i) {
        for(int j = 0; j < conv1_fp->filter.d2; ++j) {
            conv1_fp->filter(i, j) = conv1->filter(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < conv1_fp->bias.size; ++i) {
        conv1_fp->bias(i) = conv1->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    for(int i = 0; i < conv2_fp->filter.d1; ++i) {
        for(int j = 0; j < conv2_fp->filter.d2; ++j) {
            conv2_fp->filter(i, j) = conv2->filter(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < conv2_fp->bias.size; ++i) {
        conv2_fp->bias(i) = conv2->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    for(int i = 0; i < conv3_fp->filter.d1; ++i) {
        for(int j = 0; j < conv3_fp->filter.d2; ++j) {
            conv3_fp->filter(i, j) = conv3->filter(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < conv3_fp->bias.size; ++i) {
        conv3_fp->bias(i) = conv3->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    for(int i = 0; i < fc1_fp->weight.d1; ++i) {
        for(int j = 0; j < fc1_fp->weight.d2; ++j) {
            fc1_fp->weight(i, j) = fc1->weight(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < fc1_fp->bias.size; ++i) {
        fc1_fp->bias(i) = fc1->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    Tensor4D<double> trainImage(batchSize, 32, 32, 3);
    Tensor4D<i64> trainImage_fp(batchSize, 32, 32, 3);
    Tensor4D<double> e(batchSize, 10, 1, 1);
    Tensor4D<i64> e_fp(batchSize, 10, 1, 1);
    for(u64 i = 0; i < trainLen; i += batchSize) {
        for(u64 b = 0; b < batchSize; ++b) {
            for(u64 j = 0; j < 32; ++j) {
                for(u64 k = 0; k < 32; ++k) {
                    for(u64 l = 0; l < 3; ++l) {
                        trainImage(b, j, k, l) = dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0;
                        trainImage_fp(b, j, k, l) = (dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1ULL<<(scale));
                    }
                }
            }
        }

        model.forward(trainImage);
        model_fp.forward(trainImage_fp);
        softmax<double, 0>(model.activation, e);
        softmax<i64, scale>(model_fp.activation, e_fp);
        for(u64 b = 0; b < batchSize; ++b) {
            e(b, dataset.training_labels[i+b], 0, 0) -= (1.0/batchSize);
            e_fp(b, dataset.training_labels[i+b], 0, 0) -= (1ULL<<(scale))/batchSize;
        }
        model.backward(e);
        model_fp.backward(e_fp);
        printprogress(((double)i) / trainLen);
    }

    std::cout << "FP = \n";
    std::cout << fc1_fp->weight(0, 0) / ((double)(1ULL<<scale)) << " ";
    std::cout << fc1_fp->weight(0, 1) / ((double)(1ULL<<scale)) << " ";
    std::cout << fc1_fp->weight(0, 2) / ((double)(1ULL<<scale)) << " ";
    std::cout << fc1_fp->weight(0, 3) / ((double)(1ULL<<scale)) << std::endl;

    std::cout << conv1_fp->filter(0, 0) / ((double)(1ULL<<scale)) << " ";
    std::cout << conv1_fp->filter(0, 1) / ((double)(1ULL<<scale)) << " ";
    std::cout << conv1_fp->filter(0, 2) / ((double)(1ULL<<scale)) << " ";
    std::cout << conv1_fp->filter(0, 3) / ((double)(1ULL<<scale)) << std::endl;

    std::cout << "Float = \n";
    std::cout << fc1->weight(0, 0) << " ";
    std::cout << fc1->weight(0, 1) << " ";
    std::cout << fc1->weight(0, 2) << " ";
    std::cout << fc1->weight(0, 3) << std::endl;

    std::cout << conv1->filter(0, 0) << " ";
    std::cout << conv1->filter(0, 1) << " ";
    std::cout << conv1->filter(0, 2) << " ";
    std::cout << conv1->filter(0, 3) << std::endl;
}

void llama_test_vgg2(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaVersion::init();
    const u64 bs = 100;
    
    auto conv1 = new Conv2D<u64, scale, LlamaVersion>(3, 64, 3, 1);
    auto conv2 = new Conv2D<u64, scale, LlamaVersion>(64, 64, 3, 1);
    auto conv3 = new Conv2D<u64, scale, LlamaVersion>(64, 128, 3, 1);
    auto conv4 = new Conv2D<u64, scale, LlamaVersion>(128, 128, 3, 1);
    auto conv5 = new Conv2D<u64, scale, LlamaVersion>(128, 256, 3, 1);
    auto conv6 = new Conv2D<u64, scale, LlamaVersion>(256, 256, 3, 1);
    auto conv7 = new Conv2D<u64, scale, LlamaVersion>(256, 256, 3, 1);
    auto conv8 = new Conv2D<u64, scale, LlamaVersion>(256, 512, 3, 1);
    auto conv9 = new Conv2D<u64, scale, LlamaVersion>(512, 512, 3, 1);
    auto conv10 = new Conv2D<u64, scale, LlamaVersion>(512, 512, 3, 1);
    auto conv11 = new Conv2D<u64, scale, LlamaVersion>(512, 512, 3, 1);
    auto conv12 = new Conv2D<u64, scale, LlamaVersion>(512, 512, 3, 1);
    auto conv13 = new Conv2D<u64, scale, LlamaVersion>(512, 512, 3, 1);
    auto fc1 = new FC<u64, scale, LlamaVersion>(512, 256);
    auto fc2 = new FC<u64, scale, LlamaVersion>(256, 256);
    auto fc3 = new FC<u64, scale, LlamaVersion>(256, 10);
    auto model = Sequential<u64>({
        conv1,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv2,
        new SumPool2D<u64, scale, LlamaVersion>(2, 0, 2),
        new ReLUTruncate<u64, LlamaVersion>(scale+2),
        conv3,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv4,
        new SumPool2D<u64, scale, LlamaVersion>(2, 0, 2),
        new ReLUTruncate<u64, LlamaVersion>(scale+2),
        conv5,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv6,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv7,
        new SumPool2D<u64, scale, LlamaVersion>(2, 0, 2),
        new ReLUTruncate<u64, LlamaVersion>(scale+2),
        conv8,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv9,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv10,
        new SumPool2D<u64, scale, LlamaVersion>(2, 0, 2),
        new ReLUTruncate<u64, LlamaVersion>(scale+2),
        conv11,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv12,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        conv13,
        new SumPool2D<u64, scale, LlamaVersion>(2, 0, 2),
        new ReLUTruncate<u64, LlamaVersion>(scale+2),
        new Flatten<u64>(),
        fc1,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        fc2,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        fc3,
        new Truncate<u64, LlamaVersion>(scale),
    });

    auto conv1_ct = new Conv2D<i64, scale>(3, 64, 3, 1);
    conv1_ct->filter.copy(conv1->filter);
    conv1_ct->bias.copy(conv1->bias);
    auto conv2_ct = new Conv2D<i64, scale>(64, 64, 3, 1);
    conv2_ct->filter.copy(conv2->filter);
    conv2_ct->bias.copy(conv2->bias);
    auto conv3_ct = new Conv2D<i64, scale>(64, 128, 3, 1);
    conv3_ct->filter.copy(conv3->filter);
    conv3_ct->bias.copy(conv3->bias);
    auto conv4_ct = new Conv2D<i64, scale>(128, 128, 3, 1);
    conv4_ct->filter.copy(conv4->filter);
    conv4_ct->bias.copy(conv4->bias);
    auto conv5_ct = new Conv2D<i64, scale>(128, 256, 3, 1);
    conv5_ct->filter.copy(conv5->filter);
    conv5_ct->bias.copy(conv5->bias);
    auto conv6_ct = new Conv2D<i64, scale>(256, 256, 3, 1);
    conv6_ct->filter.copy(conv6->filter);
    conv6_ct->bias.copy(conv6->bias);
    auto conv7_ct = new Conv2D<i64, scale>(256, 256, 3, 1);
    conv7_ct->filter.copy(conv7->filter);
    conv7_ct->bias.copy(conv7->bias);
    auto conv8_ct = new Conv2D<i64, scale>(256, 512, 3, 1);
    conv8_ct->filter.copy(conv8->filter);
    conv8_ct->bias.copy(conv8->bias);
    auto conv9_ct = new Conv2D<i64, scale>(512, 512, 3, 1);
    conv9_ct->filter.copy(conv9->filter);
    conv9_ct->bias.copy(conv9->bias);
    auto conv10_ct = new Conv2D<i64, scale>(512, 512, 3, 1);
    conv10_ct->filter.copy(conv10->filter);
    conv10_ct->bias.copy(conv10->bias);
    auto conv11_ct = new Conv2D<i64, scale>(512, 512, 3, 1);
    conv11_ct->filter.copy(conv11->filter);
    conv11_ct->bias.copy(conv11->bias);
    auto conv12_ct = new Conv2D<i64, scale>(512, 512, 3, 1);
    conv12_ct->filter.copy(conv12->filter);
    conv12_ct->bias.copy(conv12->bias);
    auto conv13_ct = new Conv2D<i64, scale>(512, 512, 3, 1);
    conv13_ct->filter.copy(conv13->filter);
    conv13_ct->bias.copy(conv13->bias);
    auto fc1_ct = new FC<i64, scale>(512, 256);
    fc1_ct->weight.copy(fc1->weight);
    fc1_ct->bias.copy(fc1->bias);
    auto fc2_ct = new FC<i64, scale>(256, 256);
    fc2_ct->weight.copy(fc2->weight);
    fc2_ct->bias.copy(fc2->bias);
    auto fc3_ct = new FC<i64, scale>(256, 10);
    fc3_ct->weight.copy(fc3->weight);
    fc3_ct->bias.copy(fc3->bias);
    auto model_ct = Sequential<i64>({
        conv1_ct,
        new ReLUTruncate<i64>(scale),
        conv2_ct,
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        conv3_ct,
        new ReLUTruncate<i64>(scale),
        conv4_ct,
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        conv5_ct,
        new ReLUTruncate<i64>(scale),
        conv6_ct,
        new ReLUTruncate<i64>(scale),
        conv7_ct,
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        conv8_ct,
        new ReLUTruncate<i64>(scale),
        conv9_ct,
        new ReLUTruncate<i64>(scale),
        conv10_ct,
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        conv11_ct,
        new ReLUTruncate<i64>(scale),
        conv12_ct,
        new ReLUTruncate<i64>(scale),
        conv13_ct,
        new SumPool2D<i64, scale>(2, 0, 2),
        new ReLUTruncate<i64>(scale+2),
        new Flatten<i64>(),
        fc1_ct,
        new ReLUTruncate<i64>(scale),
        fc2_ct,
        new ReLUTruncate<i64>(scale),
        fc3_ct,
        new Truncate<i64>(scale),
    });

    // Tensor4D<u64> trainImage(2, 1, 2, 1); // 1 images with server and 1 with client
    Tensor4D<u64> trainImage(bs, 32, 32, 3); // 1 images with server and 1 with client
    trainImage.fill((1ULL<<(scale+1)));
    Tensor4D<i64> trainImage_ct(bs, 32, 32, 3);
    trainImage_ct.copy(trainImage);
    Tensor4D<u64> e(bs, 10, 1, 1); // 1 images with server and 1 with client
    Tensor4D<i64> e_ct(bs, 10, 1, 1);

    LlamaVersion::initializeWeights(model); // dealer initializes the weights and sends to the parties
    LlamaVersion::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    pirhana_softmax(model.activation, e, scale);
    model.backward(e);
    EndComputation();
    // LlamaVersion::output(rt->drelu);
    // LlamaVersion::output(model.activation);
    // LlamaVersion::output(e);
    // LlamaVersion::output(conv1->bias);
    if (LlamaConfig::party != 1) {
        // rt->drelu.print();
        // std::cout << "Secure Computation Output = \n";
        // model.activation.print<i64>();
        // e.print<i64>(); // eprint hehe
        // fc->filter.print<i64>();
        // conv1->bias.print<i64>();
    }
    LlamaVersion::finalize();
}
