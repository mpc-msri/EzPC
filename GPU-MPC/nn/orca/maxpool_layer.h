#pragma once

#include "utils/gpu_comms.h"

#include "fss/dcf/gpu_maxpool.h"
#include "fss/gpu_and.h"

#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {
        template <typename T>
        class MaxPool2DLayer : public GPULayer<T>
        {
        public:
            MaxpoolParams p;
            GPUMaxpoolKey<T> maxpoolKey;
            GPUSelectKey<T> backpropSelectKey;
            u32 *oneHot;
            u8 *oneHotOutputMask;
            AESGlobalContext *gaes;

            MaxPool2DLayer(int bin, int bout, int bwBackprop, int N, int imgH, int imgW, int C, int FH, int FW, int strideH,
                           int strideW, int zPadHLeft, int zPadHRight,
                           int zPadWLeft, int zPadWRight);
            ~MaxPool2DLayer();
            T *genForwardKey(uint8_t **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *g);
            T *genBackwardKey(uint8_t **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *g, int epoch);
            void readForwardKey(uint8_t **key_as_bytes);
            void readBackwardKey(uint8_t **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch);
        };
    }
}

#include "maxpool_layer.cu"
