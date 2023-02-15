#ifndef GPU_STATS_H_
#define GPU_STATS_H_

struct Stats {
    uint64_t transfer_time;
    uint64_t compute_time;
    uint64_t comm_time;
};

#endif