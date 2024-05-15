#pragma once

#include "gpu_data_types.h"
#include <cstdint>
#include <string>

extern size_t OneGB;

using u8 = uint8_t;

u8 *readFile(std::string filename, size_t *input_size, bool pin = true);
void readKey(int fd, size_t keySize, u8 *key_as_bytes, uint64_t *time);
void writeKey(std::string filename, u8 *startPtr, u8 *curPtr);
void getKeyBuf(u8 **startPtr, u8 **curPtr, size_t bufSize, bool pin = true);
void writeKeyBuf(int fd, size_t keySize, const u8 *key_as_bytes);
void getAlignedBuf(u8 **startPtr, size_t bufSize, bool pin = true);
int openForReading(std::string filename);
int openForWriting(std::string filename);
void closeFile(int fd);
void makeDir(std::string dirName);
