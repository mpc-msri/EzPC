#include "gpu_data_types.h"
#include <cstdlib>
#include <cassert>
#include <stdio.h>



// might want to double check that the integer kernel is properly written
// and that 'shared' is actually the entire matrix dimension
__global__ void embed_group_elements(GPUGroupElement *A, double *A_1, double *A_2, double *A_3, double *A_4, int N)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        GPUGroupElement mask = (static_cast<GPUGroupElement>(1) << 16) - 1;
        A_1[j] = (double)(A[j] & mask);

        A_2[j] = (double)((A[j] >> 16) & mask);

        A_3[j] = (double)((A[j] >> 32) & mask);

        A_4[j] = (double)((A[j] >> 48) & mask);
    }
}

__global__ void extract_group_elements(GPUGroupElement *A, double *A_1, double *A_2, double *A_3, double *A_4, int N)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N)
    {
        GPUGroupElement G_1 = static_cast<GPUGroupElement>(A_1[j]);
        GPUGroupElement G_2 = static_cast<GPUGroupElement>(A_2[j]);
        GPUGroupElement G_3 = static_cast<GPUGroupElement>(A_3[j]);
        GPUGroupElement G_4 = static_cast<GPUGroupElement>(A_4[j]);

        A[j] = G_1 + (G_2 << 16) + (G_3 << 32) + (G_4 << 48);
    }
}

__global__ void add_group_elements(GPUGroupElement *A, GPUGroupElement *B, GPUGroupElement *C, int N)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N)
    {
        C[j] = A[j] + B[j];
    }
}

__global__ void addInPlace(GPUGroupElement* A, GPUGroupElement* B, int bw, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        A[j] = (A[j] + B[j]) & ((1ULL << bw) - 1);
    }
}

__global__ void addModN(int numBits, uint32_t* A, uint32_t* B, int numInts) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // int numLargeInts = (numBits * N - 1) / 64 + 1;
    assert(numBits == 2);
    if (j < numInts)
    {  
        uint32_t x = A[j];
        uint32_t y = B[j];
        uint32_t z = 0;
        for(int i = 0; i < 32; i += numBits) {
            uint32_t a = (x >> i) & 3;
            uint32_t b = (y >> i) & 3;
            uint32_t c = ((a + b) & 3) << i;
            z |= c;
        }
        // printf("%d: %u\n", j, z);
        A[j] = z;
        // A[j] += B[j];
    }
}


__global__ void xorBits(uint32_t *A, uint32_t* B, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        // printf("%d: %u %u\n", j, A[j], B[j]);
        A[j] ^= B[j];
        // printf("%d: %u\n", j, A[j]);
    }
}


__global__ void beaver_add_group_elements(GPUGroupElement *A, GPUGroupElement *B, GPUGroupElement *C, GPUGroupElement *D, int N, int party)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N)
    {
        D[j] += ((party == 0 ? A[j] : 0) - B[j] - C[j]);
    }
}

__global__ void check_output(GPUGroupElement *O1, GPUGroupElement *O2, GPUGroupElement *R, int N)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N)
    {
        assert(O2[thread_id] - R[thread_id] == O1[thread_id]);
        if(thread_id < 10) printf("%d %lu %lu\n", thread_id, O2[thread_id] - R[thread_id], O1[thread_id]);
    }
}

__global__ void add_pool(GPUGroupElement *A, GPUGroupElement* B, int N, int C, int H, int W, int num_elems)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elems)
    {
        // nhwc
        int t = tid;
        int n = t / (H * W * C);
        t = t % (H * W * C);
        int h = t / (W * C);
        t = t % (W * C);
        int w = t / C;
        int c = t % C;
        // (n, c, 2h, 2w)
        int h2 = 2 * h;
        int w2 = 2 * w;
        int H2 = 2 * H;
        int W2 = 2 * W;

        int u1 = n * H2 * W2 * C + h2 * W2 * C + w2 * C + c;    
        int u2 = n * H2 * W2 * C + (h2 + 1) * W2 * C + w2 * C + c;    
        int u3 = n * H2 * W2 * C + h2 * W2 * C + (w2 + 1) * C + c;    
        int u4 = n * H2 * W2 * C + (h2 + 1) * W2 * C + (w2 + 1) * C + c;    

        B[tid] = A[u1] + A[u2] + A[u3] + A[u4];
    }
}

__global__ void add_pool_backprop(GPUGroupElement *A, GPUGroupElement* B, int N, int C, int H, int W, int num_elems)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elems)
    {
        // nhwc
        int t = tid;
        int n = t / (H * W * C);
        t = t % (H * W * C);
        int h = t / (W * C);
        t = t % (W * C);
        int w = t / C;
        int c = t % C;
        // (n, c, 2h, 2w)
        int h2 = 2 * h;
        int w2 = 2 * w;
        int H2 = 2 * H;
        int W2 = 2 * W;

        int u1 = n * H2 * W2 * C + h2 * W2 * C + w2 * C + c;    
        int u2 = n * H2 * W2 * C + (h2 + 1) * W2 * C + w2 * C + c;    
        int u3 = n * H2 * W2 * C + h2 * W2 * C + (w2 + 1) * C + c;    
        int u4 = n * H2 * W2 * C + (h2 + 1) * W2 * C + (w2 + 1) * C + c;    

        B[u1] = A[tid];
        B[u2] = A[tid];
        B[u3] = A[tid];
        B[u4] = A[tid];
        // B[tid] = A[u1] + A[u2] + A[u3] + A[u4];
    }
}


__global__ void deterministic_dropout(GPUGroupElement *A, GPUGroupElement* B, int N, int C, int H, int W, int num_elems)
{
    // NHWC
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elems)
    {
        int t = tid;
        int n = t / (H * W * C);
        t = t % (H * W * C);
        int h = t / (W * C);
        t = t % (W * C);
        int w = t / C;
        int c = t % C;
        // (n, 2h, 2w, c)
        int h2 = 2 * h;
        int w2 = 2 * w;
        int H2 = 2 * H;
        int W2 = 2 * W;
        int u = n * H2 * W2 * C + h2 * W2 * C + w2 * C + c;    
        B[tid] = A[u];
    }
}


__global__ void subtractInPlace(GPUGroupElement *A, GPUGroupElement *B, int N)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        B[j] = A[j] - B[j];
    }
}

__global__ void beaver_add_group_elements(GPUGroupElement *A, GPUGroupElement *B, GPUGroupElement *C, int N, int party)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N)
    {
        C[j] += (party == 0 ? 1 : -1) * A[j] - B[j];
    }
}