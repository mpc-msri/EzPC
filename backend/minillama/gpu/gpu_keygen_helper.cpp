#include "fss.h"
#include "gpu_data_types.h"

GroupElement *init_random(int N, int output_bit_length)
{
    GroupElement *A = new GroupElement[N];
    for (int i = 0; i < N; i++)
        A[i] = random_ge(output_bit_length);
    return A;
}

GroupElement *init_with_const(int N, int bw, uint64_t C)
{
    GroupElement *A = new GroupElement[N];
    for (int i = 0; i < N; i++)
        A[i] = GroupElement(C, bw);
    return A;
}


std::pair<GroupElement *, GroupElement *> create_shares(GroupElement *A, int N)
{
    GroupElement *A_1 = new GroupElement[N];
    GroupElement *A_2 = new GroupElement[N];
    for (int i = 0; i < N; i++)
    {
        auto split_A = splitShareCommonPRNG(A[i]);
        A_1[i] = split_A.first;
        A_2[i] = split_A.second;
    }
    return std::make_pair(A_1, A_2);
}

GroupElement *init_with_ones(int N, int bw)
{
    GroupElement *A = new GroupElement[N];
    for (int i = 0; i < N; i++)
        A[i] = GroupElement(1, bw);
    return A;
}

// GroupElement *init_with_const(int N, int bw, int C)
// {
//     GroupElement *A = new GroupElement[N];
//     for (int i = 0; i < N; i++)
//         A[i] = GroupElement(C, bw);
//     return A;
// }

void writeArrayToFile(std::ostream& f, GroupElement* A, int N) {
    for(int i = 0; i < N; i++)
    {
        f.write((char*) &A[i].value, sizeof(uint64_t));
    }
}

GPUGroupElement* CPUToGPUGroupElement(GroupElement* A, int N) {
    GPUGroupElement* B = new GPUGroupElement[N];
    for(int i=0;i<N;i++) B[i] = A[i].value;
    return B;
}

std::pair<GPUGroupElement*, GPUGroupElement*> getShares(GPUGroupElement* A, int N) {
    GPUGroupElement *A0 = new GPUGroupElement[N];
    GPUGroupElement *A1 = new GPUGroupElement[N];
    for(int i = 0; i < N; i++) {
        auto shares = splitShare(GroupElement(A[i], 64));
        A0[i] = shares.first.value;
        A1[i] = shares.second.value;
    }
    return std::make_pair(A0, A1);
}

