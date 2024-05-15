/*
Original Work Copyright (c) 2018 Xiao Wang (wangxiao@gmail.com)
Modified Work Copyright (c) 2021 Microsoft Research

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

Enquiries about further applications and development opportunities are welcome.

Modified by Deevashwer Rathee
*/

#ifndef EMP_PROTOCOL_EXECUTION_H__
#define EMP_PROTOCOL_EXECUTION_H__
#include "utils/block.h"
#include "utils/constants.h"
#include <pthread.h>

namespace sci {
class ProtocolExecution {
public:
  int cur_party;
  /*
  #ifndef THREADING
          // static ProtocolExecution * prot_exec;
  #else
          static __thread ProtocolExecution * prot_exec;
  #endif
  */

  ProtocolExecution(int party = PUBLIC) : cur_party(party) {}
  virtual ~ProtocolExecution() {}
  virtual void feed(block128 *lbls, int party, const bool *b, int nel) = 0;
  virtual void reveal(bool *out, int party, const block128 *lbls, int nel) = 0;
  virtual void finalize() {}
};
} // namespace sci
// extern sci::ProtocolExecution* prot_exec;
thread_local extern sci::ProtocolExecution *prot_exec;
#endif
