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
