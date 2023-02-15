#include "gpu_data_types.h"

uint8_t *readFile(std::string filename, size_t* input_size);
void writeSecretSharesToFile(std::ostream& f1, std::ostream& f2, int bw, int N, GPUGroupElement* A);
void writePackedBitsToFile(std::ostream& f, GPUGroupElement* A, int numBits, int N);
void readKey(int fd, size_t keySize, uint8_t* key_as_bytes);

