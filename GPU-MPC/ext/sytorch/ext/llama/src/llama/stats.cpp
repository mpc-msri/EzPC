#include <llama/stats.h>
#include <iostream>
#include <fstream>

int numRounds = 0;
uint64_t eigenMicroseconds = 0;
uint64_t accumulatedInputTimeOffline = 0;
uint64_t accumulatedInputTimeOnline = 0;

uint64_t evalMicroseconds = 0;
uint64_t reconstructMicroseconds = 0;
uint64_t arsEvalMicroseconds = 0;
uint64_t convEvalMicroseconds = 0;
uint64_t reluEvalMicroseconds = 0;
uint64_t avgpoolEvalMicroseconds = 0;
uint64_t pubdivEvalMicroseconds = 0;
uint64_t argmaxEvalMicroseconds = 0;
uint64_t multEvalMicroseconds = 0;
uint64_t selectEvalMicroseconds = 0;
uint64_t dealerMicroseconds = 0;
uint64_t inputOfflineComm = 0;
uint64_t inputOnlineComm = 0;
uint64_t startTime = 0;
uint64_t secFloatComm = 0;

uint64_t convOnlineComm = 0;
uint64_t selectOnlineComm = 0;
uint64_t arsOnlineComm = 0;
uint64_t reluOnlineComm = 0;

uint64_t packTime = 0;
uint64_t unpackTime = 0;
uint64_t sendTime = 0;
uint64_t recvTime = 0;

namespace Llama
{

    std::ostream *log_output = nullptr;
    // std::ostream *log_output = &(std::cerr);

    void stat_t::print()
    {
        if (log_output != nullptr)
        {
            (*log_output) << ">> " << name << " - Start" << std::endl;
            (*log_output) << "   Key Read Time = " << keyread_time / 1000.0 << " milliseconds\n";
            (*log_output) << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
            (*log_output) << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
            (*log_output) << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
            (*log_output) << "   Online Comm = " << comm_bytes << " bytes\n";
            (*log_output) << ">> " << name << " - End" << std::endl;
        }
    }

    std::map<std::string, stat_t> stats;

    void push_stats(const stat_t &stat)
    {
        if (stats.find(stat.name) == stats.end())
        {
            stats[stat.name] = stat;
        }
        else
        {
            stats[stat.name].compute_time += stat.compute_time;
            stats[stat.name].reconstruct_time += stat.reconstruct_time;
            stats[stat.name].keyread_time += stat.keyread_time;
            stats[stat.name].comm_bytes += stat.comm_bytes;
            stats[stat.name].keysize_bytes += stat.keysize_bytes;
        }
    }

    void dump_stats_csv(const std::string &filename)
    {
        std::ofstream out(filename);
        out << "Protocol,Online Time (ms),Communication (MB), Key Size (GB)" << std::endl;
        for (auto &stat : stats)
        {
            out << stat.second.name << "," << (stat.second.compute_time + stat.second.reconstruct_time) / 1000.0 << "," << stat.second.comm_bytes / (1024.0 * 1024.0) << "," << stat.second.keysize_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;
        }
        out.close();
    }
}
