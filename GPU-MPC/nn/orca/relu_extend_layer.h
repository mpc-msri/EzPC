#pragma once

#include "utils/gpu_comms.h"
#include "fss/dcf/gpu_relu.h"
#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class ReluExtendLayer : public GPULayer<T>
        {
        public:
            int bin, bout, /*f,*/ numRelus;
            GPUReluExtendKey<T> reluExtendKey;
            u32 *drelu;
            u8 *dReluMask;
            GPUSelectKey<T> backpropSelectKey;
            // AESGlobalContext* gaes;
            // Stats s;

            ReluExtendLayer(int bin, int bout, int numRelus);
            ~ReluExtendLayer();
            T *genForwardKey(uint8_t **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *gaes);
            T *genBackwardKey(uint8_t **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *gaes, int epoch);
            void readForwardKey(uint8_t **key_as_bytes);
            void readBackwardKey(uint8_t **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch);
        };
    }
}

#include "relu_extend_layer.cu"