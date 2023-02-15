#pragma once

#include "gpu_comms.h"
#include "gpu_data_types.h"

class Layer {
    public:    
    virtual void genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* inputMask, GPUGroupElement* outputMask) = 0;
    virtual void genBackwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask) = 0;
    virtual void readForwardKey(uint8_t** key_as_bytes) = 0;
    virtual void readBackwardKey(uint8_t** key_as_bytes) = 0;
    virtual GPUGroupElement* forward(Peer *peer, int party, GPUGroupElement* d_I, AESGlobalContext* g) = 0;
    virtual GPUGroupElement* backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* g) = 0;
    virtual void initWeights(Peer* peer, int party) {}
};
