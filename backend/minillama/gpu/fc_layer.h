// #include "gpu_conv2d.h"
#include "gpu_data_types.h"
#include "gpu_truncate.h"
#include "gpu_stats.h"
#include "gpu_comms.h"
#include "layer.h"


struct MatmulParams {
    int bin, bout;
    int M, K, N;
    size_t size_X, size_W, size_Z, size_Y;
};

class FCLayer: public Layer {
    private:
        void initMatmulKey();
        void initMatmulKeydW();
        void initMatmulKeydX();
    public:
        MatmulParams p;
        GPUMatmulKey matmulKey, matmulKeydX, matmulKeydW;
        // Z = XW + Y 
        // X = N * something, which means that Y is a row vector
        GPUGroupElement *X, *W, *Y, *Vw, *Vy;
        GPUGroupElement *mask_X, *mask_W, *mask_Z, *mask_Y = NULL, *mask_dX, *mask_dW, *mask_dY = NULL, *mask_Vw, *mask_Vy = NULL;
        TruncateType tf, tb;
        GPUSignExtendKey truncateKeyZ, truncateKeydX, truncateKeyW, truncateKeyVw, truncateKeyY, truncateKeyVy;
        Stats s;
        bool useBias;
        
        FCLayer(int bin, int bout, int M, int N, int K, TruncateType tf, TruncateType tb, bool useBias);        
        void genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_X, GPUGroupElement* mask_Z);
        void genBackwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_grad, GPUGroupElement* mask_dX);
        void readForwardKey(uint8_t** key_as_bytes);
        void readBackwardKey(uint8_t** key_as_bytes);
        GPUGroupElement* forward(Peer* peer, int party, GPUGroupElement* d_I, AESGlobalContext* gaes);
        GPUGroupElement* backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* gaes);
        void initWeights(Peer* peer, int party);

};

