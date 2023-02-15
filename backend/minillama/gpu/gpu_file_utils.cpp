#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_data_types.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include "gpu_mem.h"
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <../dcf.h>
#include <chrono>

uint8_t *readFile(std::string filename, size_t* input_size)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(filename, std::ios::binary);
    // const int fileDesc = open(filename.c_str(), O_RDONLY/*| O_DIRECT*/);
    size_t size_in_bytes = std::filesystem::file_size(filename);
    *input_size = size_in_bytes;
    uint8_t *mem_bytes = cpuMalloc(size_in_bytes);
    // assert(lseek(fileDesc, 0, SEEK_SET) != -1);
    // ssize_t numRead = read(fileDesc, (void*) mem_bytes, size_in_bytes);
    // printf("%s %lu %lu %d\n", filename.c_str(), numRead, size_in_bytes, sizeof(ssize_t));
    // assert(numRead == size_in_bytes);
    // close(fileDesc);
    file.read((char*) mem_bytes, size_in_bytes);
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time to read file in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    std::cout << "File size: " << size_in_bytes << std::endl;
    return mem_bytes;
}

char *read_from_file(std::istream& f, size_t size_in_bytes)
{
    char *mem_bytes = (char*) cpuMalloc(size_in_bytes);
    f.read(mem_bytes, size_in_bytes);
    return mem_bytes;
}


void readKey(int fd, size_t keySize, uint8_t* key_as_bytes) {
    size_t chunkSize = (1ULL << 30);
    size_t bytesRead = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while(bytesRead < keySize) {
        size_t readBytes = std::min(chunkSize, keySize - bytesRead);
        ssize_t numRead = read(fd, key_as_bytes + bytesRead, readBytes);
        assert(numRead == readBytes);
        if (numRead == -1) {
            printf("errno: %d, %s\n", errno, strerror(errno));
            assert(0 && "read");
        }
        bytesRead += numRead;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time for key read: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
}
// GPUGroupElement* readInput(int party, size_t* input_size)
// {
//     uint8_t *ip_as_bytes = readFile("input" + std::to_string(party + 1) + ".dat", input_size);
//     GPUGroupElement *I = (GPUGroupElement *)ip_as_bytes;
//     return I;
// }

// GPUGroupElement* readGPUConv2DInput(std::istream& f, GPUConv2DKey k) {
//     auto bytes = read_from_file(f, k.mem_size_I);
//     return (GPUGroupElement*) bytes;
// }

// GPUGroupElement* readGPUConv2DFilter(std::istream& f, GPUConv2DKey k) {
//     auto bytes = read_from_file(f, k.mem_size_F);
//     return (GPUGroupElement*) bytes;
// }

GPUConv2DKey readGPUConv2DKey(/*std::istream& f*/ uint8_t** key_as_bytes)
{
    // auto start = std::chrono::high_resolution_clock::now();
    GPUConv2DKey k;
    std::memcpy((char*) &k, *key_as_bytes, 15 * sizeof(int));
    *key_as_bytes += 15 * sizeof(int);

    k.size_I = k.N * k.H * k.W * k.CI;
    k.size_F = k.CO * k.FH * k.FW * k.CI;
    k.OH = ((k.H - k.FH + (k.zPadHLeft + k.zPadHRight)) / k.strideH) + 1;
    k.OW = ((k.W - k.FW + (k.zPadWLeft + k.zPadWRight)) / k.strideW) + 1;
    k.size_O = k.N * k.OH * k.OW * k.CO;
    k.mem_size_I = k.size_I * sizeof(GPUGroupElement);
    k.mem_size_F = k.size_F * sizeof(GPUGroupElement);
    k.mem_size_O = k.size_O * sizeof(GPUGroupElement);

    k.I = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_I;
    k.F = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_F;
    k.O = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_O;
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // std::cout << "Time to read Conv2D key in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    return k;
}


GPUMatmulKey readGPUMatmulKey(uint8_t** key_as_bytes)
{
    // auto start = std::chrono::high_resolution_clock::now();
    GPUMatmulKey k;
    std::memcpy((char*) &k, *key_as_bytes, 5 * sizeof(int));
    *key_as_bytes += 5 * sizeof(int);
    // f.read((char*) &k, 5 * sizeof(int));
    k.size_A = k.M * k.K; 
    k.size_B = k.K * k.N;
    k.size_C = k.M * k.N;
    k.mem_size_A = k.size_A * sizeof(GPUGroupElement);
    k.mem_size_B = k.size_B * sizeof(GPUGroupElement);
    k.mem_size_C = k.size_C * sizeof(GPUGroupElement);
    // auto key_as_bytes = read_from_file(f, k.mem_size_A + k.mem_size_B + k.mem_size_C);
    k.A = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_A;
    k.B = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_B;
    k.C = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_C;

    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // std::cout << "Time to read Matmul key in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    return k;
}


void writePackedBitsToFile(std::ostream& f, GPUGroupElement* A, int numBits, int N) {
    assert(numBits == 1 || numBits == 2);
    assert(PACKING_SIZE == 32);

    int elemsPerBlock = PACKING_SIZE / numBits;
    int numBlocks = (numBits * N - 1) / PACKING_SIZE + 1;
    uint32_t mask = (numBits == 1 ? 1 : 3);
    for(int i = 0; i < numBlocks; i++) {
        uint32_t packedBits = 0;
        for(int j = 0; j < elemsPerBlock; j++) {
            int idx = i*elemsPerBlock + j;
            if(idx < N) {
                uint32_t val = static_cast<uint32_t>(A[idx]);
                // printf("%d %u\n", idx, val);
                assert(val <= mask);
                val <<= (numBits * j);
                packedBits |= val;
                // printf("%d %u %u %d %u\n", idx, val, val << (numBits * j), numBits * j, packedBits);
            }
        }
        // printf("about to write int to file %u\n", packedBits);
        f.write((char*) &packedBits, sizeof(PACK_TYPE));
        // printf("here\n");
    }
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
    if(bw == 1 || bw == 2) {
        writePackedBitsToFile(f1, A0, bw, N);
        writePackedBitsToFile(f2, A1, bw, N);
    }
    else {
        f1.write((char*) A0, N * sizeof(GroupElement));
        f2.write((char*) A1, N * sizeof(GroupElement));
    }
    delete[] A0;
    delete[] A1;
}

// GPUGroupElement* readGPUConv2DFilter(std::istream& f, GPUConv2DKey k)
// {
//     char *bytes = read_from_file(f, k.mem_size_F);
//     return (GPUGroupElement*) bytes;
// }
GPUDCFKey readGPUDCFKey(/*std::istream& f*/ uint8_t** key_as_bytes)
{
    GPUDCFKey k;
    std::memcpy((char*) &k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);
    // f.read((char*) &k, 4 * sizeof(int));
    k.size_scw = k.num_dcfs * (k.Bin + 1);
    k.size_vcw = k.num_dcfs * (k.Bin + 1) * k.out_vec_len;
    k.mem_size_scw = k.size_scw * sizeof(AESBlock);
    if(k.Bout == 1 || k.Bout == 2) {
        assert(k.out_vec_len == 1);
        k.mem_size_vcw = ((k.Bout * k.num_dcfs - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (k.Bin + 1);
    } else k.mem_size_vcw = k.size_vcw * sizeof(GPUGroupElement);

    k.scw = (AESBlock *) *key_as_bytes;
    *key_as_bytes += k.mem_size_scw;
    k.vcw = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += k.mem_size_vcw;
    return k;
}

GPUReLUTruncateKey readGPUReLUTruncateKey(/*std::istream& f*/ uint8_t** key_as_bytes)
{
    GPUReLUTruncateKey k;
    std::memcpy((char*) &k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);

    size_t size_in_bytes = k.num_rts * sizeof(GPUGroupElement);

    k.zTruncate = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.a = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.c = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d1 = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d2 = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.dcfKeyN = readGPUDCFKey(key_as_bytes);
    k.dcfKeyS = readGPUDCFKey(key_as_bytes);

    return k;
} 

GPUReLUTruncateKey readGPULocalTruncateReLUKey(uint8_t** key_as_bytes, bool fprop)
{
    GPUReLUTruncateKey k;
    std::memcpy((char*) &k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);
    printf("rt parameters: %d %d %d %d\n", k.Bin, k.Bout, k.shift, k.num_rts);
    size_t size_in_bytes = k.num_rts * sizeof(GPUGroupElement);

    k.zTruncate = NULL;

    k.b = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    if(fprop) {
        k.a = (GPUGroupElement *) *key_as_bytes;
        *key_as_bytes += size_in_bytes;
    }

    k.c = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d1 = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d2 = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    if(fprop) {
        k.a2 = (GPUGroupElement *) *key_as_bytes;
        // number of 32-bit integers * sizeof(int)
        *key_as_bytes += ((k.num_rts - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
        k.dcfKeyN = readGPUDCFKey(key_as_bytes);
    }
    return k;
} 

// GPUDReluKey readGPUDReluKey(uint8_t** key_as_bytes) {
//     GPUDReluKey k;
//     k.dcfKey = readGPUDCFKey(key_as_bytes);
//     k.dReluMask = (uint32_t *) *key_as_bytes;
//     // number of 32-bit integers * sizeof(int)
//     *key_as_bytes += ((k.dcfKey.Bout * k.dcfKey.num_dcfs - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
//     return k;
// }

void writeDCFKeyWithOneBitOutputToFile(std::ostream& f, DCFKeyPack* dcfKey, int numRelus) {
    auto start = std::chrono::high_resolution_clock::now();
    int bin = dcfKey[0].Bin;
    int bout = dcfKey[0].Bout;
    int output_vec_length = dcfKey[0].groupSize;
    f.write((char *)&bin, sizeof(int));
    f.write((char *)&bout, sizeof(int));
    f.write((char *)&numRelus, sizeof(int));
    f.write((char *)&output_vec_length, sizeof(int));

    auto scw = new osuCrypto::block[(bin + 1) * numRelus];
    
    for (int i = 0; i < bin + 1; i++)
    {
        for (int j = 0; j < numRelus; j++)
        {
            scw[i * numRelus + j] = dcfKey[j].k[i];
            // std::cout << scw[i*numRelus + j] << std::endl;
            // f.write((char *)&dcfKey[j].k[i], sizeof(osuCrypto::block));
        }
    }
    f.write((char *) scw, (bin + 1) * numRelus * sizeof(osuCrypto::block));
    auto end1 = std::chrono::high_resolution_clock::now();
    auto elapsed = end1 - start;
    std::cout << "Time for writing scw to file: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    delete[] scw;

    assert(bout == 1 || bout == 2);
    int payloadPerBlock = PACKING_SIZE / bout;
    int numBlocks = (bout * numRelus - 1) / PACKING_SIZE + 1;
    uint64_t mask = (bout == 1 ? 1ULL : 3ULL);
    PACK_TYPE* vcw = new PACK_TYPE[(bin + 1) * numBlocks];
    for (int i = 0; i < bin; i++)
    {
        for (int j = 0; j < numBlocks; j++)
        {
            PACK_TYPE packed_bits = 0;
            for(int l = 0; l < payloadPerBlock; l++) {
                uint64_t value = 0ULL;
                int idx = j*payloadPerBlock + l;
                if(idx < numRelus) value = dcfKey[idx].v[i] & mask;
                // assert(value == 0 || value == 1);
                // if(idx == 17) printf("writing vcw %d %lu\n", i, value);
                assert(value <= mask);
                value <<= (l * bout);
                packed_bits |= static_cast<PACK_TYPE>(value);
            }
            // if(j == 1) printf("packed vcw bits: %u\n", packed_bits);
            // f.write((char*) &packed_bits, sizeof(PACK_TYPE));
            vcw[i * numBlocks + j] = packed_bits;
        }
    }
    for (int j = 0; j < numBlocks; j++) 
    {
        PACK_TYPE packed_bits = 0;
        for(int l = 0; l < payloadPerBlock; l++) {
            uint64_t value = 0ULL;
            int idx = j*payloadPerBlock + l;
            if(idx < numRelus) value = dcfKey[idx].g[0] & mask;
            assert(value <= mask);
            value <<= (l * bout);
            packed_bits |= static_cast<PACK_TYPE>(value);
        }
        vcw[bin * numBlocks + j] = packed_bits;
        // f.write((char*) &packed_bits, sizeof(PACK_TYPE));
    }
    f.write((char*) vcw, (bin + 1) * numBlocks * sizeof(PACK_TYPE));
    delete [] vcw;
    auto end2 = std::chrono::high_resolution_clock::now();
    elapsed = end2 - end1;
    std::cout << "Time for writing vcw to file: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
}
