#include <llama/stats.h>

int numRounds = 0;
uint64_t eigenMicroseconds = 0;
uint64_t accumulatedInputTimeOffline = 0;
uint64_t accumulatedInputTimeOnline = 0;

uint64_t evalMicroseconds = 0;
uint64_t reconstructMicroseconds = 0;
uint64_t arsEvalMicroseconds = 0;
uint64_t matmulEvalMicroseconds = 0;
uint64_t reluEvalMicroseconds = 0;
uint64_t convEvalMicroseconds = 0;
uint64_t maxpoolEvalMicroseconds = 0;
uint64_t avgpoolEvalMicroseconds = 0;
uint64_t pubdivEvalMicroseconds = 0;
uint64_t argmaxEvalMicroseconds = 0;
uint64_t multEvalMicroseconds = 0;
uint64_t reluTruncateEvalMicroseconds = 0;
uint64_t selectEvalMicroseconds = 0;
uint64_t dealerMicroseconds = 0;
uint64_t inputOfflineComm = 0;
uint64_t inputOnlineComm = 0;
uint64_t startTime = 0;
uint64_t secFloatComm = 0;

uint64_t convOnlineComm = 0;
uint64_t matmulOnlineComm = 0;
uint64_t selectOnlineComm = 0;
uint64_t reluOnlineComm = 0;
uint64_t arsOnlineComm = 0;
uint64_t rtOnlineComm = 0;
