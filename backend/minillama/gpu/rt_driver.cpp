#include <iostream>
#include <chrono>

#include "comms.h"
// #include "fss.h"
#include "gpu_data_types.h"
#include "helper_cuda.h"

using namespace std;

int bitlength = 64;
Peer *peer;
int party = DEALER;

extern "C" GPUReLUTruncateKey readGPUReLUTruncateKey(int party);
extern "C" GPUDCFKey readGPUDCFKey(int party, char *);

// declaration, forward
extern "C" uint64_t *gpu_relu_truncate(GPUReLUTruncateKey k,
                                       int party, uint64_t *rt_input, GPURTContext *);

extern "C" GPUGroupElement *finish_relu_truncate(GPUReLUTruncateKey k,
                                                 uint32_t *, GPUGroupElement *, GPUGroupElement *, uint32_t *, GPUGroupElement *, int);

int main(int argc, char *argv[])
{
    party = atoi(argv[1]);
    if (party == 0)
        peer = waitForPeer(42002);
    else
        peer = new Peer("10.11.0.5" /*"0.0.0.0"*/, 42002);
    peer->sync();

    auto start = std::chrono::high_resolution_clock::now();
    auto k = readGPUReLUTruncateKey(party);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    printf("Finished reading RT key: %d RTs in %ld ms\n", k.num_rts, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    GPUGroupElement *rt_input = new uint64_t[k.num_rts];
    GPUGroupElement *rt_masked_input = new uint64_t[k.num_rts];
    GPUGroupElement *rout = new uint64_t[k.num_rts];

    int mem_size_input = k.num_rts * sizeof(GPUGroupElement);

    // fss_init();

    // ifstream input("rt_randomness.txt");
    ifstream input("rt_randomness.txt", ios::binary);
    input.read((char *)rt_input, mem_size_input);
    input.read((char *)rt_masked_input, mem_size_input);
    input.read((char *)rout, mem_size_input);

    start = std::chrono::high_resolution_clock::now();
    GPURTContext ctx;
    gpu_relu_truncate(k, party, rt_masked_input, &ctx);
    // cout << output_self[0] << endl;
    // // can now write to this memory since input has been consumed
    // this is only temporary i'll wrap all this in another network call
    // need to allocate memory cleanly for this
    // this should be enough to hold everything
    GPUGroupElement *h_lrs1 = k.dcfKeyN.vcw;
    uint32_t *h_drelu1 = (uint32_t *)k.dcfKeyS.vcw;
    unsigned long drelu_mem_size = (k.num_rts - 1) / 8 + 1;
    if (party == 0)
    {
        // auto start_waiting_for_recv = std::chrono::high_resolution_clock::now();
        peer->recv_batched_input(h_lrs1, k.num_rts, k.Bin);
        peer->recv_uint8_array((uint8_t *)h_drelu1, drelu_mem_size);

        send(peer->sendsocket, ctx.h_lrs0, mem_size_input, 0);
        send(peer->sendsocket, ctx.h_drelu0, drelu_mem_size, 0);

        //     auto elapsed = std::chrono::high_resolution_clock::now() - start_waiting_for_recv;
        //     int i = 0;
        //     cout << "Own share: " << output_self[i] << endl;
        //     cout << "Received from peer: " << output_peer[i] << endl;
        //     cout << ((output_self[i] + output_peer[i]) & (static_cast<uint64_t>(-1) >> (64 - k.Bout))) << endl;
        //     cout << "Time elapsed in milliseconds for recv(): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << endl;
    }
    else
    {
        send(peer->sendsocket, ctx.h_lrs0, mem_size_input, 0);
        send(peer->sendsocket, ctx.h_drelu0, drelu_mem_size, 0);

        // Need to pack bits into the second send
        peer->recv_batched_input(h_lrs1, k.num_rts, k.Bin);
        peer->recv_uint8_array((uint8_t *)h_drelu1, drelu_mem_size);
        // peer->recv_batched_input(h_drelu1, k.num_rts, k.Bin);
    }
    // printf("drelu: %llu\n", (*ctx.h_drelu0 + *h_drelu1 - static_cast<GPUGroupElement>(13766132477404802639)) & 1);
    // printf("lrs: %llu\n", *ctx.h_lrs0 + *h_lrs1 - static_cast<GPUGroupElement>(5418726845891926763));

    auto res0 = finish_relu_truncate(k, ctx.d_drelu0, ctx.d_lrs0, ctx.d_a, h_drelu1, h_lrs1, party);
    GPUGroupElement *res1 = h_lrs1;
    if (party == 0)
    {
        peer->recv_batched_input(res1, k.num_rts, k.Bin);
        send(peer->sendsocket, res0, mem_size_input, 0);
    }
    else
    {
        send(peer->sendsocket, res0, mem_size_input, 0);
        peer->recv_batched_input(res1, k.num_rts, k.Bin);
    }
    elapsed = std::chrono::high_resolution_clock::now() - start;
    cout << "Time taken for compute and communication in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << endl;

    for (int i = 0; i < k.num_rts; i++)
    {
        auto out = res0[i] + res1[i] - rout[i];
        if (rt_input[i] >= (static_cast<GPUGroupElement>(1) << (k.Bin - 1)))
        {
            if (out != 0ULL)
            {
                printf("negative test failed: %d %lu %lu\n", i, rt_input[i], out);
                return 0;
            }
        }
        else
        {
            if (out != (rt_input[i] >> k.shift))
            {
                printf("positive test failed: %d %lu %lu\n", i, rt_input[i], out);
                return 0;
            }
        }
    }
    printf("passed\n");
    return 0;
}
