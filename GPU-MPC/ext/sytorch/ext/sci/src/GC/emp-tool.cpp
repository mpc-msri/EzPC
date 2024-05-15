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

#include "GC/circuit_execution.h"
#include "GC/protocol_execution.h"

thread_local sci::ProtocolExecution *prot_exec = nullptr;
thread_local sci::CircuitExecution *circ_exec = nullptr;
/*
#ifndef THREADING
// sci::ProtocolExecution* sci::ProtocolExecution::prot_exec = nullptr;
// sci::CircuitExecution* sci::CircuitExecution::circ_exec = nullptr;
sci::ProtocolExecution* prot_exec = nullptr;
sci::CircuitExecution* circ_exec = nullptr;
#else
__thread sci::ProtocolExecution* sci::ProtocolExecution::prot_exec = nullptr;
__thread sci::CircuitExecution* sci::CircuitExecution::circ_exec = nullptr;
#endif
*/
