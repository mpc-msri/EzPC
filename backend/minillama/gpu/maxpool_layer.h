#include "gpu_comms.h"
#include "gpu_data_types.h"
#include "gpu_select.h"
#include "gpu_relu.h"
#include "gpu_and.h"
#include "layer.h"


struct MaxpoolParams {
    int bin, bout;
    int N, imgH, imgW, C; 
    int FH, FW; 
    int strideH, strideW; 
    int zPadHLeft, zPadHRight; 
    int zPadWLeft, zPadWRight;
    int H, W;
};

// struct GPUDReluKey {
//     GPUDCFKey dcfKey;
//     uint32_t* dReluMask;
// };

// struct GPU2RoundReLUKey {
//     int bin, bout, numRelus;
//     GPUDReluKey dreluKey;
//     GPUSelectKey selectKey;
// };

// struct GPUAndKey {
//     int N;
//     uint32_t *b0, *b1, *b2;
// };

struct GPUMaxpoolKey {
    GPU2RoundReLUKey* reluKey;
    GPUAndKey* andKey;
    // GPUSelectKey selectKey;
};

class MaxPool2DLayer: public Layer {
    public:
    MaxpoolParams p;
    GPUMaxpoolKey maxpoolKey;
    GPUSelectKey backpropSelectKey;
    // GPU2RoundReLUKey* maxpoolKey;
    uint32_t* oneHot;
    GPUGroupElement* oneHotOutputMask;
    AESGlobalContext* gaes;
    Stats s;

    MaxPool2DLayer(int bin, int bout, int N, int imgH, int imgW, int C, int FH, int FW, int strideH, 
                    int strideW, int zPadHLeft, int zPadHRight, 
                    int zPadWLeft, int zPadWRight);    
    void genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* inputMask, GPUGroupElement* outputMask/*, GPUGroupElement* oneHotOutputMask, GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask*/);
    void genBackwardKey(std::ostream& f1, std::ostream& f2, /*GPUGroupElement* oneHotOutputMask,*/ GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask);
    // void init(uint8_t** key_as_bytes, AESGlobalContext* g);
    void readForwardKey(uint8_t** key_as_bytes);
    void readBackwardKey(uint8_t** key_as_bytes);
    GPUGroupElement* forward(Peer *peer, int party, GPUGroupElement* d_I, AESGlobalContext* g);
    GPUGroupElement* backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* g);
    // void initBackward(uint8_t** key_as_bytes);
    // void backward(Peer *peer, int party, GPUGroupElement* d_I);
};

