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
