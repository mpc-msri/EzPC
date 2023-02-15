#include "mini_aes.h"
#include "dcf2.cpp"
#include <omp.h>


using namespace osuCrypto;


int32_t bitlength = 64;

int main(int argc, char* argv[]) {
    int num_iterations = 6;
    double sum_of_times = 0;
    int num_threads = 12;//threads_array[k];
    int dcf_instances = atoi(argv[2]);

    block *k = new block[dcf_instances * (bitlength + 1)];
    for(int i=0;i<dcf_instances * (bitlength + 1);i++) {
        k[i] = toBlock(0, 0);
    }
    // (block*) malloc(dcf_instances * (bitlength + 1) * sizeof(block));


    GroupElement *out, input(0, 64), *g, *v, output_to_print(0, 64);
    out = (GroupElement*) malloc(dcf_instances * sizeof(GroupElement));
    g = (GroupElement*) malloc(dcf_instances * sizeof(GroupElement));
    v = (GroupElement*) malloc(dcf_instances * bitlength * sizeof(GroupElement));
    for(int i=0;i<dcf_instances;i++) {
        out[i].value = 0;
        out[i].bitsize = bitlength;
        g[i].value = 0;
        g[i].bitsize = bitlength;
        for(int j=0;j<bitlength;j++) {
            v[i*bitlength+j].value = 0;
            v[i*bitlength+j].bitsize = bitlength;
        }
    }
    // // printf("boo5 %lu\n", dcf_instances * (bitlength + 1) * sizeof(block));
    // // printf("boo6\n");
    // memset(k, 0, dcf_instances * (bitlength + 1) * sizeof(block));
    // // printf("boo7\n");
    // // GroupElement *g(0, 64), *v;

    // // GroupElement v[64 * 1];
    // // for(int i=0;i<64;i++) {
    //     // v[i].value = 0;
    //     // v[i].bitsize = 64;
    // // }
    aes_init();
    for(int j = 0; j < num_iterations; j++) {
        auto start_time = omp_get_wtime();
        #pragma omp parallel for num_threads(num_threads)
        for(uint64_t i = 0; i < static_cast<uint64_t>(dcf_instances); i++)
            {
                evalDCF(64, 64, 1, 
                &out[i], // groupSize
                1, input, 
                &k[i * (bitlength + 1)], // bin + 1
                &g[i] , // groupSize
                &v[i*bitlength], false, 0, -1); // bin * groupSize
                // if(i == atoi(argv[1])) {
                //     output_to_print.value = out[i].value;
                //     printf("%lu\n", output_to_print.value);
                // }             
            }
            auto end_time = omp_get_wtime();
            sum_of_times += (j > 0 ? end_time - start_time : 0);
        }
        printf("Time in milliseconds: %lf\n", (sum_of_times / (num_iterations - 1))*1000);
        int i = atoi(argv[1]);
        output_to_print.value = out[i].value;
        printf("%lu\n", output_to_print.value);
                  
    //                 // printed = true;
                
    // block aes_in = toBlock(0, 0);
    // block aes_out = aes_enc(aes_in, 2);
    // printf("%llu %llu\n", _mm_extract_epi64(aes_out, 0), _mm_extract_epi64(aes_out, 1));
    return 0;
}
