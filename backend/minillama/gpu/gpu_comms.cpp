#include <../comms.h>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "helper_cuda.h"
#include "gpu_mem.h"
#include <chrono>
#include "gpu_stats.h"

void send_bytes(Peer* peer, const uint8_t *data, int size)
{
    assert(size == send(peer->sendsocket, data, size, 0));
    peer->bytesSent += size;
}

void recv_bytes(Peer* peer, uint8_t *data, int size)
{
    assert(size == recv(peer->recvsocket, data, size, MSG_WAITALL));
    peer->bytesReceived += size;
}


uint8_t* exchangeShares(Peer* peer, uint8_t* to_send, size_t bytes, int party, Stats* s) {
    uint8_t* to_recv = cpuMalloc(bytes);
    auto start = std::chrono::high_resolution_clock::now();  
    std::thread send_thread(&send_bytes, peer, to_send, bytes);
    std::thread recv_thread(&recv_bytes, peer, to_recv, bytes);
    send_thread.join();
    recv_thread.join();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if(s) s->comm_time += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    // std::cout << "Time to exchange shares in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " " << s->comm_time << std::endl;
    return to_recv;
}


Peer* connectToPeer(int party, std::string addr) {
    Peer* peer;
    if (party == 0)
        peer = waitForPeer(42003);
    else
        peer = new Peer(addr /*"172.31.45.173"/* "10.11.0.5" /*"0.0.0.0"*/, 42003);
    peer->sync();
    return peer;
}


