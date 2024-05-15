#include <llama/config.h>

namespace LlamaConfig {
    int bitlength = 64;
    int num_threads = 4;
    int party = 0;
    Peer *client = nullptr;
    Peer *server = nullptr;
    Peer *peer = nullptr;
    Dealer *dealer = nullptr;
    int port = 42069;
    bool stochasticRT = false;
    bool stochasticT  = false;
}
