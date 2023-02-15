#include <Eigen/Dense>
#include <omp.h>
#include "gpu_utils.h"
#include "fss.h"



void matmul_cleartext_eigen(int dim1, int dim2, int dim3) {
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_A(dim1, dim2);
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_B(dim2, dim3);
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_C(dim1, dim3);

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      eigen_A(i, j) = i + j;//Arr2DIdxRowM(inA, dim1, dim2, i, j).value;
    }
  }
  for (int i = 0; i < dim2; i++) {
    for (int j = 0; j < dim3; j++) {
      eigen_B(i, j) = i + j;//Arr2DIdxRowM(inB, dim2, dim3, i, j).value;
    }
  }
  eigen_C = eigen_A * eigen_B;
//   for (int i = 0; i < dim1; i++) {
    // for (int j = 0; j < dim3; j++) {
    //   Arr2DIdxRowM(outC, dim1, dim3, i, j).value = eigen_C(i, j);
    // }
//   }
}

int bitlength = 64;
int party = SERVER;

int main(int argc, char* argv[]) {
  /*16384, 4096, 4096 */
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    float start, elapsed_time;

    // auto start = omp_get_wtime();
    // matmul_cleartext_eigen(m, k, n);
    // auto elapsed_time = omp_get_wtime() - start;
    // printf("Time in seconds: %lf\n", elapsed_time);

    GroupElement* A = init_with_ones(m * k, 64);
    GroupElement* B = init_with_ones(m * k, 64);
    GroupElement* C = new GroupElement[m * n];

    fss_init();
    aes_init();
    auto seed = aes_enc(toBlock(0, time(NULL)), 0);
    prngShared.SetSeed(seed);

    GroupElement *mask_A = init_random(m * k, 64);
    GroupElement *mask_B = init_random(k * n, 64);
    GroupElement *mask_C = init_random(m * n, 64);

    auto matmul_key_pack = KeyGenMatMul(64, 64, m, k, n, mask_A, mask_B, mask_C);
    auto key = matmul_key_pack.first;
    
    printf("Starting computation\n");
    start = omp_get_wtime();
    matmul_eval_helper(m, k, n, A, B, C, key.a, key.b, key.c);
    elapsed_time = omp_get_wtime() - start;
    printf("Time in seconds: %lf\n", elapsed_time);

    return 0;
}