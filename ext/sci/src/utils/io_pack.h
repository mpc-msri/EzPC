#ifndef IO_PACK_H__
#define IO_PACK_H__
#include "utils/net_io_channel.h"

#define GC_PORT_OFFSET 100
#define REV_PORT_OFFSET 50

namespace sci {
class IOPack {
public:
  NetIO *io;
  NetIO *io_rev;
  NetIO *io_GC;
  std::string address;
  int party, port;

  IOPack(int party, int port, std::string address = "127.0.0.1") {
    this->party = party;
    this->port = port;
    this->io =
        new NetIO(party == 1 ? nullptr : address.c_str(), port, false, false);
    this->io_rev = new NetIO(party == 1 ? nullptr : address.c_str(),
                             port + REV_PORT_OFFSET, false, true);
    this->io_GC = new NetIO(party == 1 ? nullptr : address.c_str(),
                            port + GC_PORT_OFFSET, true, true);
    this->io_GC->flush();
  }

  uint64_t get_rounds() {
    // no need to count io_rev->num_rounds as io_rev is only used in parallel
    // with io
    return io->num_rounds + io_GC->num_rounds;
  }

  uint64_t get_comm() { return io->counter + io_rev->counter + io_GC->counter; }

  ~IOPack() {
    delete io;
    delete io_rev;
    delete io_GC;
  }
};
} // namespace sci
#endif // IO_PACK_H__
