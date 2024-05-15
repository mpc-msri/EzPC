#include <sytorch/tensor.h>
#include <library_float.h>
#include <llama/stats.h>
#include <llama/api.h>


void secfloat_init(int secfloatParty, std::string secfloatAddr)
{
    __party = secfloatParty;
    __address = secfloatAddr;
    __init(0, nullptr);
}

void softmax_secfloat(Tensor4D<u64> &in, Tensor4D<u64> &out, u64 scale, int llamaParty)
{
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == 1);
    assert(in.d4 == 1);
    assert(out.d3 == 1);
    assert(out.d4 == 1);

    Tensor4D<u64> inFloat(in.d1, in.d2, 4, 1);
    // This hack only works when last layer is truncation layer, which is usually the case
    int origBitlength = LlamaConfig::bitlength;
    LlamaConfig::bitlength = origBitlength - scale;
    FixToFloat(in.d1 * in.d2, in.data, inFloat.data, scale);
    // printf("inFloat=%lu, %lu, %lu, %lu\n", inFloat.data[0], inFloat.data[1], inFloat.data[2], inFloat.data[3]);
    LlamaConfig::bitlength = origBitlength;
    Tensor4D<u64> outFloat(in.d1, in.d2, 4, 1);
    outFloat.fill(0);
    if (llamaParty != 1) {
        vector < vector < FPArray > > inpFloatSecfloat = make_vector_float(llamaParty-1, in.d1, in.d2);
        for(int i = 0; i < in.d1; ++i) {
            for(int j = 0; j < in.d2; ++j) {
                inpFloatSecfloat[i][j].m[0] = inFloat(i, j, 0, 0);
                inpFloatSecfloat[i][j].e[0] = inFloat(i, j, 1, 0);
                inpFloatSecfloat[i][j].z[0] = inFloat(i, j, 2, 0);
                inpFloatSecfloat[i][j].s[0] = inFloat(i, j, 3, 0);
            }
        }
        vector < vector < FPArray > > outFloatSecfloat = make_vector_float(llamaParty-1, in.d1, in.d2);

        // std::cerr << ">> Softmax (SecFloat) - Start" << std::endl;
        auto secfloat_start = std::chrono::high_resolution_clock::now();
        auto secfloat_comm_start = __get_comm();
        auto secfloat_round_start = __iopack->get_rounds();

        Softmax2(in.d1, in.d2, inpFloatSecfloat, outFloatSecfloat);
        int sz = in.d1 * in.d2;
        vector < FPArray > outFloatSecfloatFlat = make_vector_float(llamaParty-1, sz);
        for(int i = 0; i < in.d1; ++i) {
            for(int j = 0; j < in.d2; ++j) {
                outFloatSecfloatFlat[i * in.d2 + j].m[0] = outFloatSecfloat[i][j].m[0];
                outFloatSecfloatFlat[i * in.d2 + j].e[0] = outFloatSecfloat[i][j].e[0];
                outFloatSecfloatFlat[i * in.d2 + j].s[0] = outFloatSecfloat[i][j].s[0];
                outFloatSecfloatFlat[i * in.d2 + j].z[0] = outFloatSecfloat[i][j].z[0];
            }
        }
        vector<FPArray> divver = make_vector_float(llamaParty-1, sz) ;
        for (int i = 0 ; i < sz ; i++)
            divver[i] = __fp_op->input<float>(ALICE, sz, (float)(1.0/(float)in.d1)) ;
        ElemWiseMul(in.d1 * in.d2, outFloatSecfloatFlat, divver, outFloatSecfloatFlat);

        auto secfloat_round_end = __iopack->get_rounds();
        auto secfloat_comm_end = __get_comm();
        auto secfloat_end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(secfloat_end - secfloat_start).count();
        evalMicroseconds += eval_time;
        secFloatComm += (uint64_t)(secfloat_comm_end - secfloat_comm_start);
        numRounds += (secfloat_round_end - secfloat_round_start);
        // std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
        // std::cerr << ">> Softmax (SecFloat) - End" << std::endl;

        for(int i = 0; i < in.d1; ++i) {
            for(int j = 0; j < in.d2; ++j) {
                outFloat(i, j, 0, 0) = outFloatSecfloatFlat[i * in.d2 + j].m[0];
                outFloat(i, j, 1, 0) = outFloatSecfloatFlat[i * in.d2 + j].e[0];
                outFloat(i, j, 2, 0) = outFloatSecfloatFlat[i * in.d2 + j].z[0];
                outFloat(i, j, 3, 0) = outFloatSecfloatFlat[i * in.d2 + j].s[0];
            }
        }
    }
    // printf("Scale=%lu, %lu\n", scale, LlamaConfig::bitlength);
    FloatToFix(in.d1*in.d2, outFloat.data, out.data, scale);
}
