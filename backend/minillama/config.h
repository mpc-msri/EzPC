#pragma once

class Peer;
class Dealer;
namespace LlamaConfig {

    extern int bitlength;
    extern int num_threads;
    extern int party;
    extern Peer *client;
    extern Peer *server;
    extern Peer *peer;
    extern Dealer *dealer;
    extern int port;
    extern bool stochasticRT;
    extern bool stochasticT;
}
