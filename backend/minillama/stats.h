#pragma once

#include <cstdint>

extern int numRounds;
extern uint64_t eigenMicroseconds;
extern uint64_t accumulatedInputTimeOffline;
extern uint64_t accumulatedInputTimeOnline;

extern uint64_t evalMicroseconds;
extern uint64_t reconstructMicroseconds;
extern uint64_t arsEvalMicroseconds;
extern uint64_t matmulEvalMicroseconds;
extern uint64_t reluEvalMicroseconds;
extern uint64_t convEvalMicroseconds;
extern uint64_t maxpoolEvalMicroseconds;
extern uint64_t avgpoolEvalMicroseconds;
extern uint64_t pubdivEvalMicroseconds;
extern uint64_t argmaxEvalMicroseconds;
extern uint64_t multEvalMicroseconds;
extern uint64_t reluTruncateEvalMicroseconds;
extern uint64_t selectEvalMicroseconds;
extern uint64_t dealerMicroseconds;
extern uint64_t inputOfflineComm;
extern uint64_t inputOnlineComm;
extern uint64_t startTime;
extern uint64_t secFloatComm;

extern uint64_t convOnlineComm;
extern uint64_t matmulOnlineComm;
extern uint64_t selectOnlineComm;
extern uint64_t reluOnlineComm;
extern uint64_t arsOnlineComm;
extern uint64_t rtOnlineComm;
