// Authors: Kanav Gupta, Neha Jawalkar
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

#include <cstdint>
#include <map>
#include <string>

extern int numRounds;
extern uint64_t eigenMicroseconds;
extern uint64_t accumulatedInputTimeOffline;
extern uint64_t accumulatedInputTimeOnline;

extern uint64_t evalMicroseconds;
extern uint64_t reconstructMicroseconds;
extern uint64_t arsEvalMicroseconds;
extern uint64_t convEvalMicroseconds;
extern uint64_t reluEvalMicroseconds;
extern uint64_t avgpoolEvalMicroseconds;
extern uint64_t pubdivEvalMicroseconds;
extern uint64_t argmaxEvalMicroseconds;
extern uint64_t multEvalMicroseconds;
extern uint64_t selectEvalMicroseconds;
extern uint64_t dealerMicroseconds;
extern uint64_t inputOfflineComm;
extern uint64_t inputOnlineComm;
extern uint64_t startTime;
extern uint64_t secFloatComm;

extern uint64_t convOnlineComm;
extern uint64_t selectOnlineComm;
extern uint64_t arsOnlineComm;
extern uint64_t reluOnlineComm;

extern uint64_t packTime;
extern uint64_t unpackTime;
extern uint64_t sendTime;
extern uint64_t recvTime;

namespace Llama {
    struct stat_t {
        std::string name;

        uint64_t keyread_time;
        uint64_t compute_time;
        uint64_t reconstruct_time;

        uint64_t comm_bytes;
        uint64_t keysize_bytes;

        void print();
    };
    
    extern std::map<std::string, stat_t> stats;
    void push_stats(const stat_t &stat);

    void dump_stats_csv(const std::string &filename);
}
