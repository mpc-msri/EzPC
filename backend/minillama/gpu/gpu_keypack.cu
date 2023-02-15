#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "gpu_data_types.h"
// #include "cuda.h"
// #include "cuda_runtime.h"


char *read_from_file(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    size_t size_in_bytes = std::filesystem::file_size(filename);
    printf("file size: %lu\n", size_in_bytes);
    char *mem_bytes;
    cudaMallocHost((void**)&mem_bytes, size_in_bytes);
    file.read(mem_bytes, size_in_bytes);
    return mem_bytes;
}

extern "C" GPUConv2DKey readGPUConv2DKey(int party)
{
    char *key_as_bytes = read_from_file("conv2d_key" + std::to_string(party + 1) + ".dat");
    GPUConv2DKey k;
    int bytes_read = 15 * sizeof(int);
    memcpy(&k, key_as_bytes, bytes_read);
    k.size_I = k.N * k.H * k.W * k.CI;
    k.size_F = k.CO * k.FH * k.FW * k.CI;
    k.OH = ((k.H - k.FH + (k.zPadHLeft + k.zPadHRight)) / k.strideH) + 1;
    k.OW = ((k.W - k.FW + (k.zPadWLeft + k.zPadWRight)) / k.strideW) + 1;
    k.size_O = k.N * k.OH * k.OW * k.CO;
    k.mem_size_I = k.size_I * sizeof(GPUGroupElement);
    k.mem_size_F = k.size_F * sizeof(GPUGroupElement);
    k.mem_size_O = k.size_O * sizeof(GPUGroupElement);
    k.I = (GPUGroupElement *)&key_as_bytes[bytes_read];
    k.F = (GPUGroupElement *)&key_as_bytes[bytes_read + k.mem_size_I];
    k.O = (GPUGroupElement *)&key_as_bytes[bytes_read + k.mem_size_I + k.mem_size_F];
    return k;
}

extern "C" std::pair<GPUGroupElement *, GPUGroupElement *> readGPUConv2DInput(int party, int size_I_in_bytes)
{
    char *ip_as_bytes = read_from_file("conv2d_input" + std::to_string(party + 1) + ".dat");
    GPUGroupElement *I = (GPUGroupElement *)ip_as_bytes;
    GPUGroupElement *F = (GPUGroupElement *)&ip_as_bytes[size_I_in_bytes];
    return std::make_pair(I, F);
}

extern "C" GPUDCFKey readGPUDCFKey(int party, char *key_as_bytes)
{
    if (!key_as_bytes)
    {
        key_as_bytes = read_from_file("dcf_key" + std::to_string(party + 1) + ".dat");
    }
    GPUDCFKey k;
    int bytes_read = 4 * sizeof(int);
    memcpy(&k, key_as_bytes, bytes_read);
    k.size_scw = k.num_dcfs * (k.Bin + 1);
    k.size_vcw = k.num_dcfs * (k.Bin + 1) * k.out_vec_len;
    k.mem_size_scw = k.size_scw * sizeof(AESBlock);
    k.mem_size_vcw = k.size_vcw * sizeof(GPUGroupElement);
    k.scw = (AESBlock *)&key_as_bytes[bytes_read];
    k.vcw = (GPUGroupElement *)&key_as_bytes[bytes_read + k.mem_size_scw];
    return k;
}

extern "C" GPUReLUTruncateKey readGPUReLUTruncateKey(int party)
{
    char *key_as_bytes = read_from_file("rt_key" + std::to_string(party + 1) + ".dat");
    GPUReLUTruncateKey k;

    unsigned long int bytes_read = 4 * sizeof(int);
    memcpy(&k, key_as_bytes, bytes_read);
    key_as_bytes += bytes_read;

    bytes_read = sizeof(GPUGroupElement) * k.num_rts;

    k.zTruncate = (GPUGroupElement *)key_as_bytes;
    key_as_bytes += bytes_read;

    k.a = (GPUGroupElement *)key_as_bytes;
    key_as_bytes += bytes_read;

    k.b = (GPUGroupElement *)key_as_bytes;
    key_as_bytes += bytes_read;

    k.c = (GPUGroupElement *)key_as_bytes;
    key_as_bytes += bytes_read;

    k.d1 = (GPUGroupElement *)key_as_bytes;
    key_as_bytes += bytes_read;

    k.d2 = (GPUGroupElement *)key_as_bytes;
    key_as_bytes += bytes_read;

    k.dcfKeyN = readGPUDCFKey(party, key_as_bytes);
    bytes_read = (4 * sizeof(int) + k.dcfKeyN.mem_size_scw +
                  k.dcfKeyN.mem_size_vcw);

    key_as_bytes += bytes_read;
    k.dcfKeyS = readGPUDCFKey(party, key_as_bytes);
    return k;
} 
