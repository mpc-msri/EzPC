#include "mini_aes.h"
#include "dcf.h"
#include "fss.h"
#include <iostream>
#include <fstream>
// #include <Eigen/Dense>
#include <math.h>
#include "gpu_utils.h"

using namespace std;

int bitlength;
int party = DEALER;

pair<GroupElement *, GroupElement *> create_shares(GroupElement *A, int N)
{
    GroupElement *A_1 = new GroupElement[N];
    GroupElement *A_2 = new GroupElement[N];
    for (int i = 0; i < N; i++)
    {
        auto split_A = splitShareCommonPRNG(A[i]);
        A_1[i] = split_A.first;
        A_2[i] = split_A.second;
    }
    return make_pair(A_1, A_2);
}

GroupElement *init_with_ones(int N, int bw)
{
    GroupElement *A = new GroupElement[N];
    for (int i = 0; i < N; i++)
        A[i] = GroupElement(1, bw);
    return A;
}

GroupElement *init_with_const(int N, int bw, int C)
{
    GroupElement *A = new GroupElement[N];
    for (int i = 0; i < N; i++)
        A[i] = GroupElement(C, bw);
    return A;
}

void writeMatMulInputToFile(string filename, GroupElement *A, GroupElement *B, int N, int M)
{
    ofstream Key;

    Key.open(filename, ios::out | ios::binary);
    for (int i = 0; i < N; i++)
        Key.write((char *)&A[i].value, sizeof(uint64_t));
    for (int i = 0; i < M; i++)
        Key.write((char *)&B[i].value, sizeof(uint64_t));
    Key.close();
}

void writeDCFKeyToFile(string filename, DCFKeyPack key_pack, int dcf_instances)
{
    ofstream Key;
    Key.open(filename, ios::out | ios::binary);
    // cout << Key << endl;
    // ofstream Key(filename, ios::out | ios::binary);
    // Key.write(&key_pack.Bin << endl;
    // Key << key_pack.Bout << endl;
    // Key << key_pack.groupSize << endl;
    for (int k = 0; k < dcf_instances; k++)
    {
        for (int i = 0; i < key_pack.Bin + 1; i++)
        {
            Key.write((char *)&key_pack.k[i], sizeof(block));
        }
    }
    for (int k = 0; k < dcf_instances; k++)
    {
        for (int j = 0; j < key_pack.Bin; j++)
        {
            for (int i = 0; i < key_pack.groupSize; i++)
            {
                Key.write((char *)&key_pack.v[j * key_pack.groupSize + i].value, sizeof(uint64_t));
            }
        }
        for (int i = 0; i < key_pack.groupSize; i++)
        {
            Key.write((char *)&key_pack.g[i].value, sizeof(uint64_t));
        }
    }
    Key.close();
}

void writeMatMulKeyToFile(string filename, MatMulKey key)
{
    ofstream Key;
    Key.open(filename, ios::out | ios::binary);
    for (int i = 0; i < key.s1 * key.s2; i++)
        Key.write((char *)&key.a[i].value, sizeof(uint64_t));
    for (int i = 0; i < key.s2 * key.s3; i++)
        Key.write((char *)&key.b[i].value, sizeof(uint64_t));
    for (int i = 0; i < key.s1 * key.s3; i++)
        Key.write((char *)&key.c[i].value, sizeof(uint64_t));
    Key.close();
    printf("%lu\n", key.c[0].value);
}

// GroupElement *init_random(int N, int output_bit_length)
// {
//     GroupElement *A = new GroupElement[N];
//     for (int i = 0; i < N; i++)
//         A[i] = random_ge(output_bit_length);
//     return A;
// }
/* figure out why this is giving out only zeros */
/* try for (64, 32), (64, 40), (64, 64)
           (32, 32), (32, 40), (32, 64)
           */
int main(int argc, char *argv[])
{
    bitlength = atoi(argv[2]);
    aes_init();
    fss_init();
    auto seed = aes_enc(toBlock(0, time(NULL)), 0);
    prngShared.SetSeed(seed);

    int input_bit_length = atoi(argv[1]);  //64;//64;
    int output_bit_length = atoi(argv[2]); //32;//64;
    GroupElement g = GroupElement(14, output_bit_length);
    auto dcf_key_pack = keyGenDCF(input_bit_length, output_bit_length, 1, GroupElement(40, output_bit_length), &g);
    // printf("%d %d %d\n", key_pack.first.Bin, key_pack.second.Bout, key_pack.second.groupSize);
    int m = atoi(argv[3]), k = atoi(argv[4]), n = atoi(argv[5]); //32768; //65536;//131072; //262144;

    int dcf_instances = m * n;//atoi(argv[3]);
    // cout << key_pack.first.v[0].value << endl;
    // cout << key_pack.second.v[0].value << endl;
    // cout << key_pack.first.k[0] << endl;
    // cout << key_pack.second.k[0] << endl;
    writeDCFKeyToFile("dcf_key1_" + std::to_string(input_bit_length) + "_" + std::to_string(output_bit_length) + "_" + std::to_string(dcf_instances) + ".dat", dcf_key_pack.first, dcf_instances);
    writeDCFKeyToFile("dcf_key2_" + std::to_string(input_bit_length) + "_" + std::to_string(output_bit_length) + "_" + std::to_string(dcf_instances) + ".dat", dcf_key_pack.second, dcf_instances);
    // GroupElement x = GroupElement(0, 64);
    // GroupElement e1 = GroupElement(0, 64);
    // GroupElement e2 = GroupElement(0, 64);
    // evalDCF(0, &e1, x, key_pack.first);
    // evalDCF(1, &e2, x, key_pack.second);
    // GroupElement res0 = e1 + e2;
    // cout << e1 << endl;
    // cout << e2 << endl;
    // mod(res0);
    // cout << res0 << endl;
    // assert(res0 == g);

    // int m = 1024, k = 1024, n = 1024;
    // int m = atoi(argv[3]), k = atoi(argv[4]), n = atoi(argv[5]); //32768; //65536;//131072; //262144;
    // GroupElement *mask_A = init_random(m * k, output_bit_length);
    // GroupElement *mask_B = init_random(k * n, output_bit_length);
    // GroupElement *mask_C = init_random(m * n, output_bit_length);
    GroupElement *mask_A = init_with_const(m * k, output_bit_length, 1);
    GroupElement *mask_B = init_with_const(k * n, output_bit_length, 1);
    GroupElement *mask_C = init_with_const(m * n, output_bit_length, 0);
    // printf("C: %lu\n", mask_C[0].value);
    // init_random(output_bit_length, mask_A, m, k);
    // init_random(output_bit_length, mask_B, k, n);

    auto matmul_key_pack = KeyGenMatMul(input_bit_length, output_bit_length, m, k, n, mask_A, mask_B, mask_C);
    // printf("%lu %lu %lu\n", mask_A[0].value, mask_B[0].value, mask_C[0].value);
    // printf("%lu %lu %lu\n", matmul_key_pack.first.a[0].value, matmul_key_pack.first.b[0].value, matmul_key_pack.first.c[0].value);
    // printf("%lu %lu %lu\n", matmul_key_pack.second.a[0].value, matmul_key_pack.second.b[0].value, matmul_key_pack.second.c[0].value);
    // printf("%lu %lu %lu\n", matmul_key_pack.first.a[0].value + matmul_key_pack.second.a[0].value, matmul_key_pack.first.b[0].value + matmul_key_pack.second.b[0].value, matmul_key_pack.first.c[0].value + matmul_key_pack.second.c[0].value);

    // MatMulKey k1, k2;
    // k1.a = init_with_ones(m * k, output_bit_length);
    // k1.b = init_with_ones(n * k, output_bit_length);
    // k1.c = init_with_const(m * n, output_bit_length, 2048);

    // k2.a = init_with_ones(m * k, output_bit_length);
    // k2.b = init_with_ones(n * k, output_bit_length);
    // k2.c = init_with_const(m * n, output_bit_length, 2048);

    // k1.s1 = k2.s1 = m;
    // k1.s2 = k2.s2 = k;
    // k1.s3 = k2.s3 = n;

    // auto matmul_key_pack = std::make_pair(k1, k2);
    ofstream dims;
    dims.open("dims.dat", ios::out | ios::binary);
    dims.write((char *)&matmul_key_pack.first.s1, sizeof(int));
    dims.write((char *)&matmul_key_pack.first.s2, sizeof(int));
    dims.write((char *)&matmul_key_pack.first.s3, sizeof(int));

    writeMatMulKeyToFile("matmul_key1.dat", matmul_key_pack.first);
    writeMatMulKeyToFile("matmul_key2.dat", matmul_key_pack.second);
    GroupElement *A = init_with_ones(m * k, output_bit_length);
    GroupElement *B = init_with_ones(n * k, output_bit_length);

    auto split_A = create_shares(A, m * k);
    auto split_B = create_shares(B, n * k);
    // printf("%lu %lu\n", split_A.first[0].value, matmul_key_pack.first.a[0].value);
    // GroupElement *A_0, *A_1, *B_0, *B_1;
    // A_0 = init_with_ones(mk);
    // A_1 = init_with_ones(m, k);
    // B_0 = init_with_ones(n, k);
    // B_1 = init_with_ones(n, k);

    // auto split_A = std::make_pair(init_with_ones(m * k, output_bit_length), init_with_ones(m * k, output_bit_length));
    // auto split_B = std::make_pair(init_with_ones(n * k, output_bit_length), init_with_ones(n * k, output_bit_length));
    // printf("boo %lu\n", split_A.first);

    writeMatMulInputToFile("matmul_input1.dat", split_A.first, split_B.first, m * k, n * k);
    writeMatMulInputToFile("matmul_input2.dat", split_A.second, split_B.second, m * k, n * k);
}
/* There appear to be two prng objects -- one is called prng and the other
is called sharedprng. i need to initialize at least sharedprng before generating
keys for matmul */
