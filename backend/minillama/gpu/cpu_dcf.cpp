#include "mini_aes.h"
#include "dcf2.cpp"
#include <omp.h>
#include <fstream>


using namespace std;


int32_t bitlength = 32;

int main(int argc, char* argv[]) {
    int num_iterations = 6;
    double sum_of_times = 0;
    int num_threads = 12;
    int dcf_instances = atoi(argv[2]);
    int party = 0;

    int num_blocks = dcf_instances * (bitlength + 1);
    int block_size_in_bytes = num_blocks * sizeof(osuCrypto::block);

   osuCrypto::block *k = new block[num_blocks];

   string key_filename = "key" + std::to_string(party + 1) + "_" + std::to_string(dcf_instances) + ".dat";

    ifstream dcf_key(key_filename, ios::out | ios::binary);

    dcf_key.read((char *) k, block_size_in_bytes);

    // for(int i=0;i<dcf_instances * (bitlength + 1);i++) {
    //     k[i] = toBlock(0, 0);
    // }

    GroupElement *out, input(0, 32), *g, *v, output_to_print(0, 32);
    out = (GroupElement*) malloc(dcf_instances * sizeof(GroupElement));
    g = (GroupElement*) malloc(dcf_instances * sizeof(GroupElement));
    v = (GroupElement*) malloc(dcf_instances * bitlength * sizeof(GroupElement));
    for(int i=0;i<dcf_instances;i++) {
        for(int j=0;j<bitlength;j++) {
            dcf_key.read((char *) &v[i*bitlength+j].value, sizeof(uint64_t));
            v[i*bitlength+j].bitsize = bitlength;
        }
        // out[i].value = 0;
        // out[i].bitsize = bitlength;
    }
    for(int i=0;i<dcf_instances;i++) {
        dcf_key.read((char *) &g[i].value, sizeof(uint64_t));
        g[i].bitsize = bitlength;
    }


    aes_init();
    for(int j = 0; j < num_iterations; j++) {
        for(int i=0;i<dcf_instances;i++) {
            out[i].value = 0;
            out[i].bitsize = bitlength;
        }
        auto start_time = omp_get_wtime();
        #pragma omp parallel for num_threads(num_threads)
        for(uint64_t i = 0; i < static_cast<uint64_t>(dcf_instances); i++)
            {
                evalDCF(32, 32, 1, 
                &out[i], // groupSize
                0, input, 
                &k[i * (bitlength + 1)], // bin + 1
                &g[i] , // groupSize
                &v[i*bitlength], false, 0, -1); // bin * groupSize
            }
            auto end_time = omp_get_wtime();
            sum_of_times += (j > 0 ? end_time - start_time : 0);
        }
        printf("Time in milliseconds: %lf\n", (sum_of_times / (num_iterations - 1))*1000);
        int i = atoi(argv[1]);
        // output_to_print.value = out[i].value;
        cout << out[i] << endl;
        // printf("%lu\n", mod(output_to_print));
                  
    //                 // printed = true;
                
    // block aes_in = toBlock(0, 0);
    // block aes_out = aes_enc(aes_in, 2);
    // printf("%llu %llu\n", _mm_extract_epi64(aes_out, 0), _mm_extract_epi64(aes_out, 1));
    return 0;
}
