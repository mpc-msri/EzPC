#include <iostream>
#include <chrono>

#include "comms.h"
#include "gpu_data_types.h"

using namespace std;

int bitlength = 64; // = 64;
Peer *peer;
int party = DEALER;

extern "C" GPUDCFKey readGPUDCFKey(int party);

// declaration, forward
extern "C" uint64_t *gpu_dcf(GPUDCFKey k,
                             int party, uint64_t *dcf_input);

int main(int argc, char *argv[])
{
    party = atoi(argv[1]);
    auto k = readGPUDCFKey(party);
    printf("Finished reading DCF key: %d\n", k.num_dcfs);
    if (party == 0)
        peer = waitForPeer(42002);
    else
        peer = new Peer("10.11.0.5"/*"0.0.0.0"*/, 42002);
    peer->sync();

    uint64_t *dcf_input = new uint64_t[k.num_dcfs];
    int mem_size_input = k.num_dcfs * sizeof(GPUGroupElement);

    memset(dcf_input, 0, sizeof(dcf_input));

    auto start = std::chrono::high_resolution_clock::now();
    auto output_self = gpu_dcf(k, party, dcf_input);
    // can now write to this memory since input has been consumed
    auto output_peer = dcf_input;
    if (party == 0)
    {
        // uint64_t* peers_share = new uint64_t[k.num_dcfs];
        auto start_waiting_for_recv = std::chrono::high_resolution_clock::now();
        peer->recv_batched_input(output_peer, k.num_dcfs, k.Bout);
        send(peer->sendsocket, output_self, mem_size_input, 0);
        auto elapsed = std::chrono::high_resolution_clock::now() - start_waiting_for_recv;
        int i = 187;
        cout << "Own share: " << output_self[i] << endl;
        cout << "Received from peer: " << output_peer[i] << endl;
        cout << ((output_self[i] + output_peer[i]) & (static_cast<uint64_t>(-1) >> (64 - k.Bout))) << endl;
        cout << "Time elapsed in milliseconds for recv(): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << endl;
    }
    else
    {
        send(peer->sendsocket, output_self, mem_size_input, 0);
        // cout << "Bytes sent: " << peer->bytesSent << endl;
        peer->recv_batched_input(output_peer, k.num_dcfs, k.Bout);
    }
}
