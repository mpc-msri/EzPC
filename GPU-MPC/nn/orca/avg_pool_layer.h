#pragma once

#include "utils/gpu_comms.h"
#include "fss/dcf/gpu_truncate.h"
#include "nn/orca/gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class AvgPool2DLayer : public GPULayer<T>
        {

        public:
            AvgPoolParams p;
            dcf::TruncateType tf, tb;
            dcf::GPUTruncateKey<T> truncateKeyF, truncateKeyB;
            int inSz, outSz;

            AvgPool2DLayer(int bin, int bout, int scaleDiv, int N, int imgH, int imgW, int C, int FH, int FW, int strideH,
                           int strideW, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, TruncateType tf, TruncateType tb);
            T *genForwardKey(uint8_t **key_as_bytes, int party, T *inputMask, AESGlobalContext *g);
            T *genBackwardKey(uint8_t **key_as_bytes, int party, T *incomingGradMask, AESGlobalContext *g, int epoch);
            void readForwardKey(uint8_t **key_as_bytes);
            void readBackwardKey(uint8_t **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch);
        };
    }
}

#include "avg_pool_layer.cu"