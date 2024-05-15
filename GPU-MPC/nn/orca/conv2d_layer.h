#pragma once

#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_conv2d.h"
#include "fss/dcf/gpu_truncate.h"

#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class Conv2DLayer : public GPULayer<T>
        {

        private:
            void initConvKey();
            void initConvKeydI();
            void initConvKeydF();

        public:
            Conv2DParams p;
            GPUConv2DKey<T> convKey, convKeydI, convKeydF;
            bool inputIsShares;
            T *I, *F, *Vf, *b, *Vb;
            // Stats s;
            bool useBias, computedI;
            TruncateType tf, tb;
            GPUTruncateKey<T> truncateKeyC, truncateKeydI, truncateKeyF, truncateKeyVf, truncateKeyb, truncateKeyVb;

            // using these variables for keygen
            T *mask_I, *d_mask_I, *mask_F, *mask_Vf, *mask_b, *mask_Vb;

            Conv2DLayer(int bin, int bout, int N, int H, int W, int CI, int FH, int FW, int CO,
                        int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, bool useBias, dcf::TruncateType tf, dcf::TruncateType tb, bool computedI, bool inputIsShares);
            T *genForwardKey(u8 **key_as_bytes, int party, T *mask_I, AESGlobalContext *gaes);
            T *genBackwardKey(u8 **key_as_bytes, int party, T *mask_grad, AESGlobalContext *gaes, int epoch);
            void readForwardKey(u8 **key_as_bytes);
            void readBackwardKey(u8 **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *gaes, int epoch);
            void initWeights(u8 **weights, bool floatWeights);
            void dumpWeights(std::ofstream &f);
        };
    }
}
#include "conv2d_layer.cu"
