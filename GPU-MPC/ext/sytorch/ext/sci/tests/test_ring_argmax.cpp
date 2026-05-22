/*
Authors: Mayank Rathee, Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "NonLinear/argmax.h"

using namespace std;
using namespace sci;

int party = 0;
int32_t bitlength = 32;
int num_threads = 1;
int port = 32000;
string address = "127.0.0.1";
int num_argmax = 1000;

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.arg("N", num_argmax, "Number of elements");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  IOPack *iopack = new IOPack(party, port, address);
  uint64_t magnitude_bound = (1 << (bitlength - 3));
  uint64_t mask_l = -1ULL;
  if (bitlength != 64) {
    mask_l = (1ULL << bitlength) - 1ULL;
  }
  OTPack otpack(iopack, party);
  ArgMaxProtocol<uint64_t> argmax_oracle(party, RING, iopack, bitlength, MILL_PARAM, 0, &otpack);
  PRG128 prg;
  uint64_t *input_share1, *input_share2, *input_share_uncorrected;
  uint8_t *input_share_sign;
  input_share_uncorrected = new uint64_t[num_argmax];
  input_share1 = new uint64_t[num_argmax];
  input_share2 = new uint64_t[num_argmax];
  input_share_sign = new uint8_t[num_argmax];

  uint64_t *argmax_output_protocol = new uint64_t[1];
  uint64_t *argmax_output_protocol_share_other = new uint64_t[1];
  uint64_t *argmax_output_protocol_arg = new uint64_t[1];
  uint64_t *argmax_output_protocol_share_other_arg = new uint64_t[1];
  uint64_t *argmax_output_actual = new uint64_t[1];
  switch (party) {
  case ALICE: {
    prg.random_data(input_share_uncorrected, sizeof(uint64_t) * num_argmax);
    prg.random_data(input_share_sign, num_argmax);
    uint64_t comm_start = iopack->get_comm();
    auto start = clock_start();
    for (int i = 0; i < num_argmax; i++) {
      input_share_uncorrected[i] %= magnitude_bound;
    }
    for (int i = 0; i < num_argmax; i++) {
      if (input_share_sign[i] & 1) {
        input_share1[i] = -1 * input_share_uncorrected[i];
      } else {
        input_share1[i] = input_share_uncorrected[i];
      }
    }
    argmax_oracle.ArgMaxMPC(num_argmax, input_share1,
                            argmax_output_protocol_arg, true,
                            argmax_output_protocol);

    long long t = time_from(start);
    uint64_t comm_end = iopack->get_comm();
    cout << "Comparison Time\t" << RED << (double(num_argmax) / t) * 1e6
         << " ArgMax/sec" << RESET << endl;
    cout << "ALICE communication\t" << BLUE
         << ((double)(comm_end - comm_start) * 8) / (bitlength * num_argmax) << "*" << bitlength
         << " bits/ArgMax" << RESET << endl;
    std::cout << "ALICE: Done MaxPool protocol execution" << std::endl;
    iopack->io->recv_data(input_share2, sizeof(uint64_t) * num_argmax);
    iopack->io->recv_data(argmax_output_protocol_share_other,
                          sizeof(uint64_t) * 1);
    iopack->io->recv_data(argmax_output_protocol_share_other_arg,
                          sizeof(uint64_t) * 1);

    cout << "Checking correctness of ArgMax now..." << endl;

    argmax_output_protocol[0] =
        (argmax_output_protocol[0] + argmax_output_protocol_share_other[0]) &
        mask_l;
    argmax_output_protocol_arg[0] =
        (argmax_output_protocol_arg[0] +
         argmax_output_protocol_share_other_arg[0]) &
        mask_l;

    for (int i = 0; i < num_argmax; i++) {
      input_share1[i] = input_share1[i] + input_share2[i];
    }

    argmax_output_actual[0] = input_share1[0];
    for (int i = 1; i < num_argmax; i++) {
      argmax_output_actual[0] = (((argmax_output_actual[0] - input_share1[i]) >=
                                  (1 << (bitlength - 1)))
                                     ? input_share1[i]
                                     : argmax_output_actual[0]);
    }
    std::cout << "Max Protocol: " << argmax_output_protocol[0] << std::endl;
    std::cout << "Max Actual: " << argmax_output_actual[0] << std::endl;
    std::cout << "ArgMax Protocol: " << argmax_output_protocol_arg[0]
              << std::endl;

    assert(argmax_output_actual[0] == argmax_output_protocol[0] &&
           "ArgMax output is incorrect");

    cout << "ArgMax answer is: " << GREEN << "CORRECT!" << RESET << endl;
    break;
  }
  case BOB: {
    // These are written so that overall time excludes these.
    prg.random_data(input_share_uncorrected, sizeof(uint64_t) * num_argmax);
    prg.random_data(input_share_sign, num_argmax);
    uint64_t comm_start = iopack->get_comm();

    for (int i = 0; i < num_argmax; i++) {
      input_share_uncorrected[i] %= magnitude_bound;
    }
    for (int i = 0; i < num_argmax; i++) {
      if (input_share_sign[i] & 1) {
        input_share2[i] = -1 * input_share_uncorrected[i];
      } else {
        input_share2[i] = input_share_uncorrected[i];
      }
    }
    argmax_oracle.ArgMaxMPC(num_argmax, input_share2,
                            argmax_output_protocol_arg, true,
                            argmax_output_protocol);

    uint64_t comm_end = iopack->get_comm();
    cout << "BOB communication\t" << BLUE
         << ((double)(comm_end - comm_start) * 8) / (bitlength * num_argmax) << "*" << bitlength
         << " bits/ArgMax" << RESET << endl;

    std::cout << "BOB: Done MaxPool protocol execution" << std::endl;
    iopack->io->send_data(input_share2, sizeof(uint64_t) * num_argmax);
    iopack->io->send_data(argmax_output_protocol, sizeof(uint64_t) * 1);
    iopack->io->send_data(argmax_output_protocol_arg, sizeof(uint64_t) * 1);
    break;
  }
  }
  return 0;
}
