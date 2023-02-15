#include <../comms.h>
#include <thread>
#include "gpu_stats.h"

void send_bytes(Peer* peer, const uint8_t *data, int size);
void recv_bytes(Peer* peer, uint8_t *data, int size);
uint8_t* exchangeShares(Peer* p, uint8_t* to_send, size_t bytes, int party, Stats* s);
Peer* connectToPeer(int party, std::string addr);



