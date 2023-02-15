// #include "gpu_conv2d.h"
#include "gpu_data_types.h"
#include "gpu_truncate.h"
#include "gpu_stats.h"
#include "gpu_comms.h"
#include "layer.h"


struct Conv2DParams {
    int bin, bout, N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight,
        zPadWLeft, zPadWRight,
        strideH, strideW, OH, OW;
    size_t size_I, size_F, size_O;
    // size_t mem_size_I, mem_size_F, mem_size_O;    
    // GPUGroupElement *I, *F, *O;

};

// struct GPUConv2DKey
// {
//     int Bin, Bout, N, H, W, CI, FH, FW, CO,
//         zPadHLeft, zPadHRight,
//         zPadWLeft, zPadWRight,
//         strideH, strideW, OH, OW;
//     size_t size_I, size_F, size_O;
//     size_t mem_size_I, mem_size_F, mem_size_O;    
//     GPUGroupElement *I, *F, *O;
// };



class Conv2DLayer: public Layer {
    private: 
        // this is for updating F and b values
        // GPUGroupElement *mask_updated_F, *mask_updated_Vf, *mask_updated_b, *mask_updated_Vb;

        void initConvKey();
        void initConvKeydI();
        void initConvKeydF();
        // void truncate(TruncateType t, int shift, Peer* peer, int party, GPUGroupElement* d_I);
    public:
        Conv2DParams p;
        GPUConv2DKey convKey, convKeydI, convKeydF;
        GPUGroupElement *I, *F, *Vf, *b, *Vb;
        Stats s;
        bool useBias, computedI;
        TruncateType tf, tb;
        GPUSignExtendKey truncateKeyC, truncateKeydI, truncateKeyF, truncateKeyVf, truncateKeyb, truncateKeyVb;
        
        // using these variables for keygen
        GPUGroupElement *mask_I, *d_mask_I,/**mask_C, *mask_dI,*/ *mask_F, /**mask_dF,*/ *mask_Vf, *mask_b, /**mask_db,*/ *mask_Vb;


        Conv2DLayer(int bin, int bout, int N, int H, int W, int CI, int FH, int FW, int CO, 
        int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, bool useBias, TruncateType tf, TruncateType tb, bool computedI);
        // void clear();
        void genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_I, GPUGroupElement* mask_C);
        void genBackwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_grad, GPUGroupElement* mask_dI);
        void readForwardKey(uint8_t** key_as_bytes);
        void readBackwardKey(uint8_t** key_as_bytes);
        GPUGroupElement* forward(Peer* peer, int party, GPUGroupElement* d_I, AESGlobalContext* gaes);
        GPUGroupElement* backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* gaes);
        void initWeights(Peer* peer, int party);
};

