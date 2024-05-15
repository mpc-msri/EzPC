#pragma once

#include "utils/gpu_comms.h"

#include "fss/gpu_select.h"
#include "fss/gpu_and.h"
#include "fss/dcf/gpu_relu.h"

#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class ReluLayer : public GPULayer<T>
        {
        public:
            int bin, bout, numRelus;
            GPU2RoundReLUKey<T> reluKey;
            GPUSelectKey<T> backpropSelectKey;
            u32 *drelu;
            u8 *dReluMask;

            ReluLayer(int bin, int bout, int numRelus);
            T *genForwardKey(uint8_t **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *g);
            T *genBackwardKey(uint8_t **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *g, int epoch);
            void readForwardKey(uint8_t **key_as_bytes);
            void readBackwardKey(uint8_t **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch);
        };
    }
}

#include "relu_layer.cu"