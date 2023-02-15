#include "fss.h"
#include "gpu_data_types.h"

GroupElement *init_random(int N, int output_bit_length);
GroupElement *init_with_const(int N, int bw, uint64_t C);
std::pair<GroupElement *, GroupElement *> create_shares(GroupElement *A, int N);
GroupElement *init_with_ones(int N, int bw);
void writeArrayToFile(std::ostream& f, GroupElement* A, int N);
GPUGroupElement* CPUToGPUGroupElement(GroupElement* A, int N);
std::pair<GPUGroupElement*, GPUGroupElement*> getShares(GPUGroupElement* A, int N);
