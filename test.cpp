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
