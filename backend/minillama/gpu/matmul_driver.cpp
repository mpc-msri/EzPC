#include <iostream>
#include <chrono>

#include "comms.h"
// #include "gpu_utils.h"

using namespace std;

int bitlength; // = 64;
Peer *peer;
int party;

// declaration, forward
extern "C" pair<uint64_t *, uint64_t *> gpu_matmul(int input_bit_len,
                                                   int output_bit_len, int m, int n, int k,
                                                   int party);
extern "C" uint64_t* finish_gpu_matmul(int input_bit_len,
                                                          int output_bit_len, int m, int n, int k,
                                                          int party, uint64_t *A, uint64_t *B);

int main(int argc, char *argv[])
{
    party = atoi(argv[1]);
    int input_bit_length = atoi(argv[2]);
    int output_bit_length = atoi(argv[3]);
    // int m, n, k;
    // m = n = k = 1024;
    std::ifstream dims("dims.dat", std::ios::out | std::ios::binary);
    int m, k, n;
    // int m = 4096, k = 16384, n = 128; //32768; //65536;//131072; //262144;
    dims.read((char *)&m, sizeof(int));
    dims.read((char *)&k, sizeof(int));
    dims.read((char *)&n, sizeof(int));

    printf("dims: %d %d %d\n",m, k, n);

    if (party == 0)
        peer = waitForPeer(42002);
    else
        peer = new Peer("10.11.0.5"/*"0.0.0.0"*/, 42002);
    peer->sync();
    auto start = std::chrono::high_resolution_clock::now();
    auto masked_input = gpu_matmul(input_bit_length, output_bit_length, m, n, k, party);
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed1 = stop - start;
    cout << "Computation 1: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed1).count() << endl;
    uint64_t *A_plus_mask_A = new uint64_t[m * k];
    uint64_t *B_plus_mask_B = new uint64_t[n * k];
    if (party == 0)
    {
        peer->recv_batched_input(A_plus_mask_A, m * k, output_bit_length);
        peer->recv_batched_input(B_plus_mask_B, n * k, output_bit_length);
        send(peer->sendsocket, masked_input.first, 8 * m * k, 0);
        send(peer->sendsocket, masked_input.second, 8 * n * k, 0);
    }
    else
    {
        send(peer->sendsocket, masked_input.first, 8 * m * k, 0);
        send(peer->sendsocket, masked_input.second, 8 * n * k, 0);
        peer->recv_batched_input(A_plus_mask_A, m * k, output_bit_length);
        peer->recv_batched_input(B_plus_mask_B, n * k, output_bit_length);
    }
    start = std::chrono::high_resolution_clock::now();
    uint64_t *share_C_0 = finish_gpu_matmul(input_bit_length, output_bit_length, m, n, k, party, A_plus_mask_A, B_plus_mask_B);
    stop = std::chrono::high_resolution_clock::now();
    auto elapsed2 = stop - start;
    cout << "Computation 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed2).count() << endl;
    cout << "Total: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed1 + elapsed2).count() << endl;
    uint64_t *share_C_1 = new uint64_t[m * n];
    /* receive input and combine it */
    if (party == 0)
    {
        /* this receive can run entirely in parallel with the gpu computation */
        peer->recv_batched_input(share_C_1, m * n, output_bit_length);
        send(peer->sendsocket, share_C_0, 8 * m * n, 0);
    } else {
        send(peer->sendsocket, share_C_0, 8 * m * n, 0);
        peer->recv_batched_input(share_C_1, m * n, output_bit_length);
    }
    printf("%lu %lu %lu\n", share_C_0[0], share_C_1[0], share_C_0[0] + share_C_1[0]);
}
