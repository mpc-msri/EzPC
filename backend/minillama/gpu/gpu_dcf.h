#include "gpu_data_types.h"
#include "../dcf.h"

// uint8_t *readFile(std::string filename, size_t* input_size);
GPUDCFKey readGPUDCFKey(/*std::istream& f*/ uint8_t** key_as_bytes);
void writeDCFKeyWithOneBitOutputToFile(std::ostream& f, DCFKeyPack* k, int numRelus);
std::pair<DCFKeyPack, DCFKeyPack> cpuKeyGenDCF(int Bin, int Bout, GroupElement idx, GroupElement payload);