#include <fstream>
#include <cassert>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "gpu_data.h"


Dataset readDataset(std::string name, int party)
{
    Dataset d;
    if (name.compare("cifar10") == 0)
    {
        d.images = 50000;
        d.H = 32;
        d.W = 32;
        d.C = 3;
        d.classes = 10;
    }
    else if (name.compare("mnist") == 0)
    {
        d.images = 60000;
        d.H = 28;
        d.W = 28;
        d.C = 1;
        d.classes = 10;
    }
    else if (name.compare("imagenet") == 0)
    {
        d.images = 16;
        d.H = 224;
        d.W = 224;
        d.C = 3;
        d.classes = 1000;
    }
    else if (name.compare("320x320x3") == 0)
    {
        d.images = 16;
        d.H = 320;
        d.W = 320;
        d.C = 3;
        d.classes = 14;
    }
    else
    {
        assert(0 && "nothing matched!");
    }
    u64 *data, *labels;
    size_t dataSize, labelSize;
    if (name.compare("imagenet") == 0 || name.compare("320x320x3") == 0)
    {
        dataSize = d.images * d.H * d.W * d.C;
        data = (u64 *)cpuMalloc(dataSize * sizeof(u64));
        randomGEOnCpu(dataSize, 64, data);
        labelSize = d.images * d.classes;
        labels = (u64 *)cpuMalloc(labelSize * sizeof(u64));
        randomGEOnCpu(labelSize, 64, labels);
    }
    else
    {
        auto sharesDir = "./datasets/shares/" + name + "/";
        data = (u64 *)readFile(sharesDir + name + "_share" + std::to_string(party + 1) + ".dat", &dataSize);
        labels = (u64 *)readFile(sharesDir + name + "_labels" + std::to_string(party + 1) + ".dat", &labelSize);
    }
    d.data = data;
    d.labels = labels;
    return d;
}
