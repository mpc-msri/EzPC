#pragma once

#include "utils/gpu_data_types.h"

struct Dataset {
    int images;
    int H, W, C;
    int classes;
    u64* data;
    u64* labels;
};

Dataset readDataset(std::string name, int party);

#include "gpu_data.cu"