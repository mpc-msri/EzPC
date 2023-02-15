#include "fss.h"
#include "gpu_keygen_helper.h"

int getConv2DInputSize(int N, int H, int W, int CI);
int getConv2DFilterSize(int CI, int FH, int FW, int CO);
int getConv2DOutputSize(int N, int H, int W, int CI, int FH, int FW, int CO,
int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight,
int strideH, int strideW);
GPUGroupElement* getFilter(int CO, int FH, int FW, int CI, 
GroupElement* f1, int size_f);
GroupElement* init_random(int size, int bw);
GroupElement* gpuWriteKeyConv2D(std::ostream&, std::ostream&,
 int, int, int, int, int, int, int, int, int, GroupElement*, 
 int, GroupElement*, int, int&);
void gpuWriteKeyConv2DBackProp(std::ostream& f1, std::ostream& f2, 
int bin, int bout, int N, int H, int W, int CI, int FH, int FW, int CO,
GroupElement* mask_grad_cpu, int size_grad, GroupElement* mask_I_cpu, int size_I, GroupElement* mask_F_cpu, int size_F,
GroupElement* mask_dI_cpu, GroupElement* mask_dF_cpu);
GroupElement* gpuWriteKeyReluTruncate(std::ostream&, std::ostream&, int, int, int, GroupElement*, 
int);
std::pair<GroupElement*, GroupElement*> gpuWriteKeyLocalTruncateRelu(std::ostream&, std::ostream&, int, int, int, GroupElement*, 
int);
GroupElement* gpuWriteKeyLocalTruncateReluBackProp(std::ostream&, std::ostream&, int, int, int, GroupElement*, GroupElement*,
int);
GroupElement* gpuWriteKeyRelu(std::ostream&, std::ostream&, int, int, int, GroupElement*, 
int);
GroupElement* gpuWriteKeyMatmul(std::ostream&, std::ostream&,
 int, int, int, int, int, GroupElement*, GroupElement*, int&);
void writeArrayToFile(std::ostream& f, GroupElement* A, int N);
GroupElement* gpuWriteKeyMatmulBackProp(std::ostream& f1, std::ostream& f2, 
int bin, int bout, int M, int N, int K,
GroupElement* mask_X_cpu, GroupElement* mask_W_cpu, GroupElement* mask_grad);
 