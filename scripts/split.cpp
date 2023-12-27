#include "../cifar10.hpp"

using GPUGroupElement = uint64_t;

inline void mod(GPUGroupElement &a, int bw)
{
    if (bw != 64)
        a = a & ((uint64_t(1) << bw) - 1); 
}

inline std::pair<GPUGroupElement, GPUGroupElement> splitShare(const GPUGroupElement& a, int bw)
{
    GPUGroupElement a1, a2;
    a1 = rand();
    // a1 = 0;
    mod(a1, bw);
    a2 = a - a1;
    mod(a2, bw);
    return std::make_pair(a1, a2);
}

void writeSecretSharesToFile(std::ostream& f1, std::ostream& f2, int bw, int N, GPUGroupElement* A) {
    GPUGroupElement* A0 = new GPUGroupElement[N];
    GPUGroupElement* A1 = new GPUGroupElement[N];
    for(int i = 0; i < N; i++) {
        auto shares = splitShare(A[i], bw);
        A0[i] = shares.first;
        A1[i] = shares.second;
        // printf("%lu --> %lu %lu\n", A[i], A0[i], A1[i]);
    }

    f1.write((char*) A0, N * sizeof(GPUGroupElement));
    f2.write((char*) A1, N * sizeof(GPUGroupElement));
    delete[] A0;
    delete[] A1;
}

std::pair<GPUGroupElement*, int> readCifar10Labels() {
    int N = 50000;
    GPUGroupElement* data = new GPUGroupElement[N * 10];
    auto rawData = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        uint64_t label = rawData.training_labels[i];
        data[10 * i + label] = 1;
        for(int j = 0; j < 10; j++) {
            if(j != label) {
                data[10 * i + j] = 0;
            }
        }
    } 
    return std::make_pair(data, N * 10);
}

std::pair<GPUGroupElement*, int> readCifar10(int scale) {
    int N = 50000, H = 32, W = 32, C = 3;
    GPUGroupElement* data = new GPUGroupElement[N * H * W * C];
    // if(party == 0) {
        auto rawData = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < H; j++) {
                for(int k = 0; k < W; k++) {
                    for(int l = 0; l < C; l++) {
                        data[i * H * W * C + j * W * C + k * C + l] = (rawData.training_images[i][l * H * W + j * W + k] / 255.0) * (1LL << (scale));
                    }
                }
            }
        }
    // } 
    // else {
        // assert(party == 1);
        // memset(data, 0, N * H * W * C * sizeof(GPUGroupElement));
    // }    
    return std::make_pair(data, N * H * W * C);
}

void shareCifar10(int bw) {
    auto res = readCifar10Labels();
    auto data = res.first;
    auto N = res.second;
    std::ofstream f1("cifar10_labels1.dat"), f2("cifar10_labels2.dat"); 
    writeSecretSharesToFile(f1, f2, bw, N, data);
}

void shareCifar10Data(int bw) {
    auto res = readCifar10(24);
    auto data = res.first;
    auto N = res.second;
    std::ofstream f1("cifar10_share1.dat"), f2("cifar10_share2.dat"); 
    writeSecretSharesToFile(f1, f2, bw, N, data);
}

int main() {
    shareCifar10(64);
    shareCifar10Data(64);
    return 0;
}