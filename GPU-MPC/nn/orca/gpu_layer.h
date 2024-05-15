#pragma once

#include "fss/gpu_aes_shm.h"
#include "utils/gpu_comms.h"
#include <cassert>
#include <vector>
// #include <mutex>
// #include <condition_variable>
namespace dcf
{
    namespace orca
    {

        template <typename T>
        class GPULayer
        {
        public:
            std::string name = "";
            bool train = false;
            bool useMomentum = true;
            bool loadedWeights = false;
            Stats s;
            virtual void setTrain(bool useMomentum)
            {
                train = true;
                this->useMomentum = useMomentum;
            }
            void checkIfTrain()
            {
                assert(train && "train is not set!");
            }

            virtual T *genForwardKey(u8 **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *g) = 0;
            virtual T *genBackwardKey(u8 **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *g, int epoch) = 0;
            virtual void readForwardKey(u8 **key_as_bytes) = 0;
            virtual void readBackwardKey(u8 **key_as_bytes, int epoch) = 0;
            virtual T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g) = 0;
            virtual T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch) = 0;
            virtual void initWeights(u8 **weights, bool floatWeights) {}
            virtual void dumpWeights(std::ofstream &f) {}
        };

    }
}