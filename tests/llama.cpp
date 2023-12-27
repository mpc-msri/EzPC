
#include "../backend/minillama/relu.h"
#include "../backend/minillama/and.h"
#include "../backend/minillama/api.h"
#include "../utils.h"
#include "../backend/llama.h"
#include "../backend/llama_extended.h"
#include "../softmax.h"
#include "../layers.h"

void pt_test_maxpooldouble()
{
    auto r1 = random_ge(64);
    auto r2 = random_ge(64);
    auto rbit = random_ge(1);
    auto rout = random_ge(64);
    auto keys = keyGenMaxpoolDouble(64, 64, r1, r2, rbit, rout);
    auto x = 66 + r1;
    auto y = 77 + r2;
    auto b0 = evalMaxpoolDouble_1(0, x, y, keys.first);
    auto b1 = evalMaxpoolDouble_1(1, x, y, keys.second);
    auto b = b0 + b1;
    // always_assert((b % 2) == ((1 + rbit) % 2));
    std::cout << (b - rbit) % 2 << std::endl;
    auto max0 = evalMaxpoolDouble_2(0, x, y, b, keys.first);
    auto max1 = evalMaxpoolDouble_2(1, x, y, b, keys.second);
    auto max = max0 + max1;
    // always_assert(max == ((x > y) ? x : y) + rout);
    std::cout << (max - rout) << std::endl;
}

void pt_test_bitwiseand()
{
    auto r1 = random_ge(64);
    auto r2 = random_ge(64);
    auto rout = random_ge(64);
    auto keys = keyGenBitwiseAnd(r1, r2, rout);
    auto x = 66 + r1;
    auto y = 77 + r2;
    auto b0 = evalBitwiseAnd(0, x, y, keys.first);
    auto b1 = evalBitwiseAnd(1, x, y, keys.second);
    auto b = b0 ^ b1;
    // always_assert(b == ((66 & 77) ^ rout));
    std::cout << (b ^ rout) << std::endl;
    std::cout << (66 & 77) << std::endl;
}


void llama_fixtofloat_test(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    // LlamaExtended<u64>::init("172.31.45.158");
    LlamaExtended<u64>::init("127.0.0.1", true);
    int numClasses = 10;
    int batchSize = 100;
    if (party != 1)
        // secfloat_init(party - 1, "172.31.45.158");
        secfloat_init(party - 1, "127.0.0.1");
    
    Tensor4D<u64> e(batchSize, numClasses, 1, 1);
    e.fill(0);
    Tensor4D<u64> y(batchSize, numClasses, 1, 1);
    
    for(int i = 0; i < numClasses; ++i) {
        e(0, i, 0, 0) = i * (1ULL << scale);
        e(1, i, 0, 0) = 5 * (1ULL << scale);
    }
    
    if (LlamaConfig::party == 1) {
        e.fill(0);
    }
    y.fill(0);

    // LlamaExtended<u64>::initializeData(e, 1);
    StartComputation();
    softmax_secfloat(e, y, scale, party);
    EndComputation();
    Llama<u64>::output(y);
    LlamaExtended<i64>::finalize();
    if (LlamaConfig::party != 1) {
        y.print();
    }
    // LlamaExtended<u64>::finalize();
    // for(int i = 0; i < y.d1; ++i) {
    //     for(int j = 0; j < y.d2; ++j) {
    //         y(i, j, 0, 0) = y(i, j, 0, 0) % (1ULL << 24);
    //         y(i, j, 2, 0) = y(i, j, 2, 0) % 2;
    //         y(i, j, 3, 0) = y(i, j, 3, 0) % 2;
    //     }
    // }
    // if (LlamaConfig::party != 1) {
    //     y.print<i64>();
    // }
}

void relu_real_keysize() {
    using LlamaVersion = LlamaExtended<u64>;
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 16;
    LlamaConfig::party = 1;
    LlamaVersion::init("172.31.45.173");
    const u64 bs = 1;

    auto model = Sequential<u64>({
        new ReLU<u64, LlamaVersion>(),
    });

    Tensor4D<u64> trainImage(bs, 1, 1, 1);
    trainImage.fill(0);

    StartComputation();
    model.forward(trainImage);
    EndComputation();
    LlamaVersion::finalize();
}


void llama_test_vgg2(int party) {
    // dont forget to change `useLocalTruncation` to `true` in respective backend
    using LlamaVersion = Llama<u64>;
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaVersion::init("172.31.45.173");
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
    LlamaVersion::output(model.activation);
    // LlamaVersion::output(e);
    LlamaVersion::output(conv1->bias);
    if (LlamaConfig::party != 1) {
        // rt->drelu.print();
        std::cout << "Secure Computation Output = \n";
        model.activation.print<i64>();
        // e.print<i64>(); // eprint hehe
        // fc->filter.print<i64>();
        conv1->bias.print<i64>();
    }
    LlamaVersion::finalize();

    if (LlamaConfig::party == 1) {
        std::cout << "Cleartext Computation Output = \n";
        model_ct.forward(trainImage_ct);
        pirhana_softmax_ct(model_ct.activation, e_ct, scale);
        model_ct.backward(e_ct);
        model_ct.activation.print<i64>();
        conv1_ct->bias.print<i64>();
    }
}
