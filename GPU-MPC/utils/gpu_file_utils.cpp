// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdint>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <chrono>

#include "helper_cuda.h"
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "gpu_file_utils.h"

extern int errno;

// using u8 = uint8_t;

u8 *readFile(std::string filename, size_t *input_size, bool pin)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(filename, std::ios::binary);
    size_t size_in_bytes = std::filesystem::file_size(filename);
    *input_size = size_in_bytes;
    u8 *mem_bytes = cpuMalloc(size_in_bytes, pin);
    file.read((char *)mem_bytes, size_in_bytes);
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    // std::cout << "Time to read file in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    // std::cout << "File size: " << size_in_bytes << std::endl;
    return mem_bytes;
}

void readKey(int fd, size_t keySize, u8 *key_as_bytes, uint64_t *time)
{
    size_t chunkSize = (1ULL << 30);
    size_t bytesRead = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (bytesRead < keySize)
    {
        auto start = std::chrono::high_resolution_clock::now();
        size_t toRead = std::min(chunkSize, keySize - bytesRead);
        ssize_t numRead = read(fd, key_as_bytes + bytesRead, toRead);
        if (numRead == -1)
        {
            printf("errno: %d, %s\n", errno, strerror(errno));
            assert(0 && "read");
        }
        assert(numRead == toRead);
        bytesRead += numRead;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        // std::cout << "Time for key read: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (time)
        *time += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    std::cout << "Time for key read: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
}

void writeKeyBuf(int fd, size_t keySize, const u8 *key_as_bytes)
{
    size_t chunkSize = (1ULL << 30);
    size_t bytesWritten = 0;
    // printf("%d, %lu, %lx\n", fd, keySize, key_as_bytes);
    // auto start = std::chrono::high_resolution_clock::now();
    while (bytesWritten < keySize)
    {
        size_t toWrite = std::min(chunkSize, keySize - bytesWritten);
        // printf("%d, %lx, %lu\n", fd, key_as_bytes + bytesWritten, toWrite);
        ssize_t numWritten = write(fd, key_as_bytes + bytesWritten, toWrite);
        if (numWritten == -1)
        {
            printf("errno: %d, %s\n", errno, strerror(errno));
            assert(0 && "write");
        }
        assert(numWritten == toWrite);
        bytesWritten += numWritten;
    }
    // sync();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // std::cout << "Time for key write: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
}

int openForReading(std::string filename)
{
    printf("Opening file=%s\n", filename.data());
    int fd = open(filename.data(), O_RDONLY | O_DIRECT | O_LARGEFILE);
    if (fd == -1)
        assert(0 && "fopen");
    lseek(fd, 0, SEEK_SET);
    return fd;
}

int openForWriting(std::string filename)
{
    int fd = open(filename.data(), O_WRONLY | O_DIRECT | O_LARGEFILE | O_TRUNC | O_CREAT, 0644);
    if (fd == -1)
        assert(0 && "fopen");
    return fd;
}

void writeKey(std::string filename, u8 *startPtr, u8 *curPtr)
{
    std::ofstream f(filename);
    size_t keyBytes = curPtr - startPtr;
    f.write((char *)startPtr, keyBytes);
    f.close();
}

void getKeyBuf(u8 **startPtr, u8 **curPtr, size_t bufSize, bool pin)
{
    // printf("Getting key buf\n");
    *startPtr = cpuMalloc(/*4 * OneGB*/ bufSize, pin);
    *curPtr = *startPtr;
}

void getAlignedBuf(u8 **startPtr, size_t bufSize, bool pin)
{
    int err = posix_memalign((void **)startPtr, 4096, bufSize);
    assert(err == 0 && "posix memalign");
    if (pin)
        checkCudaErrors(cudaHostRegister(*startPtr, bufSize, cudaHostRegisterDefault));
}

void closeFile(int fd)
{
    int error = close(fd);
    assert(error == 0 && "close file");
}

void makeDir(std::string dirName)
{
    if (!std::filesystem::create_directory(dirName))
    {
        if (errno == EEXIST)
        {
            return;
        }
        else
        {
            assert(0 && "could not create directory");
        }
    }
}
