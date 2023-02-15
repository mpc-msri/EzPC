#include "gpu_comms.h"
#include "gpu_data_types.h"
// #include "gpu_select.h"
#include "gpu_relu.h"
#include "layer.h"


struct GPUReluSignExtendKey {
    GPUDReluKey dReluKey;
    uint32_t* dcfMask;
    GPUGroupElement* oneHot;
    GPUGroupElement* outMask;
};

class ReluSignExtendLayer: public Layer {
    public:
    int bin, bout, /*f,*/ numRelus;
    GPUReluSignExtendKey reluSignExtendKey;
    uint32_t* drelu;
    GPUGroupElement* dReluMask;
    GPUSelectKey backpropSelectKey;
    // AESGlobalContext* gaes;
    Stats s;

    ReluSignExtendLayer(int bin, int bout, int numRelus);    
    void genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* inputMask, /*GPUGroupElement* dReluMask, GPUGroupElement* dcfMask,*/ GPUGroupElement* outputMask /*, GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask*/);
    void genBackwardKey(std::ostream& f1, std::ostream& f2, /*GPUGroupElement* inputMask, GPUGroupElement* dReluMask, GPUGroupElement* dcfMask, GPUGroupElement* outputMask,*/ GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask);
    void readForwardKey(uint8_t** key_as_bytes);
    void readBackwardKey(uint8_t** key_as_bytes);
    GPUGroupElement* forward(Peer *peer, int party, GPUGroupElement* d_I, AESGlobalContext* g);
    GPUGroupElement* backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* g);
    // void initBackward(uint8_t** key_as_bytes);
    // void backward(Peer *peer, int party, GPUGroupElement* d_I);
};
