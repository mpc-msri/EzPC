/*

Authors: Aseem Rastogi, Nishant Kumar, Mayank Rathee.

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

#ifndef __EZPC_H_
#define __EZPC_H_
#include "millionaire_prob.h"
#include "../../../abycore/circuit/booleancircuits.h"
#include "../../../abycore/sharing/sharing.h"
#include<vector>
#include<string>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<iostream>
#include<cmath>
using namespace std;

/*
 * somehow we need this redirection for adding Cons gates
 * directly calling PutConsGate gives an error
 */
share* put_cons32_gate(Circuit* c, uint32_t val) {
  uint32_t x = val;
  return c->PutCONSGate(x, (uint32_t)32);
}

share* put_cons64_gate(Circuit* c, uint64_t val) {
  uint64_t x = val;
  return c->PutCONSGate(x, (uint32_t)64);
}

share* put_cons1_gate(Circuit* c, uint64_t val) {
  uint64_t x = val;
  return c->PutCONSGate(x, (uint32_t)1);
}

share* left_shift(Circuit* c, share* val, uint32_t shift_factor) {
  uint32_t share_wire_count = val->get_bitlength();
  share* fresh_zero_share = put_cons32_gate(c, 0);
  std::vector<uint32_t> val_wires = val->get_wires();
  if(share_wire_count == 1){
    cout<<"Error. Share not padded. A share cannot exist with just 1 wire.\n";
  }
  // Note here the assumption is that if we receive the val share as a share of size 32, we output the share as a share of size 32 only and drop the MSBs which overflow the 32 bit constraint.
  for(int i=0; i+shift_factor<share_wire_count; i++){
    fresh_zero_share->set_wire_id(shift_factor+i, val_wires[i]);
  }
  return fresh_zero_share;
}

share* get_zero_share(Circuit* c, int bitlen){
  if (bitlen == 32)
    return put_cons32_gate(c, 0);
  else
    return put_cons64_gate(c, 0);
}

share* logical_right_shift(Circuit* c, share* val, uint32_t shift_factor) {
  int bitlen = val->get_bitlength();
  vector<uint32_t> val_wires = val->get_wires();

  vector<uint32_t> zero_share_wires = (get_zero_share(c, bitlen))->get_wires();
  vector<uint32_t> new_val_wires(bitlen, 0);
  for(int i=0; i<bitlen; i++){
    if (i >= (bitlen - shift_factor)){
      new_val_wires[i] = zero_share_wires[i];
    }
    else{
      new_val_wires[i] = val_wires[i+shift_factor];
    }
  }
  share* x = create_new_share(new_val_wires, c);
  return x;
}

share* arithmetic_right_shift(Circuit* c, share* val, uint32_t shift_factor) {
  int bitlen = val->get_bitlength();
  share* neg_val = c->PutSUBGate(get_zero_share(c, bitlen), val);
  share* is_pos = c->PutGTGate(neg_val, val);
  val = c->PutMUXGate(val, neg_val, is_pos);
  share* x = logical_right_shift(c, val, shift_factor);
  return c->PutMUXGate(x, c->PutSUBGate(get_zero_share(c, bitlen), x), is_pos);
}

/*
 * we maintain a queue of outputs
 * basically every OUTPUT adds an OUTGate,
 * and adds the returned share to this queue
 * this queue is then flushed at the end after we have done exec
 */

struct output_queue_elmt {
  ostream& os;  //output stream to which we will output (cout or files), can this be a reference to prevent copying?
  e_role role;  //who should we output the clear value to
  enum {PrintMsg, PrintValue } kind;
  string msg;
  share *ptr;
};

typedef  vector<output_queue_elmt> output_queue;
/*
 * called from the EzPC generated code
 */
void add_to_output_queue(output_queue &q,
			 share *ptr,
			 e_role role,
			 ostream &os)
{
  struct output_queue_elmt elmt { os, role, output_queue_elmt::PrintValue, "", ptr };
  q.push_back(elmt);
}

void add_print_msg_to_output_queue (output_queue &q, string msg, e_role role, ostream &os)
{
  struct output_queue_elmt elmt { os, role, output_queue_elmt::PrintMsg, msg, NULL };
  q.push_back(elmt); 
}

/*
 * flush the queue
 * both parties call this function with their role
 * called from the EzPC generated code
 */
void flush_output_queue(output_queue &q, e_role role, uint32_t bitlen)
{
  for(output_queue::iterator it = q.begin(); it != q.end(); ++it) {  //iterate over the queue
    if (it->kind == output_queue_elmt::PrintValue) {
      if(it->role == ALL || it->role == role) {  //if the queue element role is same as mine
        if(bitlen == 32) {  //output to the stream
          it->os << it->ptr->get_clear_value<uint32_t>() << endl;
        } else {
          it->os << it->ptr->get_clear_value<uint64_t>() << endl;
        }
      }
    } else {
      if(it->role == ALL || it->role == role) {  //if the queue element role is same as mine
        it->os << it->msg << endl;
      }
    }
  }
}

/*
 * function to write a single share
 * called from EzPC generated code
 *
 * TODO: confused about bitlen
 */
void write_share(share *ptr, Circuit *circ, uint32_t bitlen, e_role role,
		 ofstream &of_add,
		 ofstream &of_rand,
		 output_queue &q)
{
  /* input shares of a random value, SERVER handles rand, CLIENT handles added shares */
  share *rand_sh = role == SERVER ? circ->PutINGate((uint32_t)rand(), bitlen, SERVER) : circ->PutDummyINGate(bitlen);
  /* add the input share with the random share */
  share *add_sh = circ->PutADDGate(ptr, rand_sh);
  /* add to the output q, so that it gets written out */
  add_to_output_queue(q, circ->PutOUTGate(rand_sh, SERVER), SERVER, of_rand);
  add_to_output_queue(q, circ->PutOUTGate(add_sh, CLIENT), CLIENT, of_add);
  /* TODO: can we optimize OUTPUT gates for the random value, since we already have its clear value in hand? */
  return;
}

template<typename T>
share *read_share(Circuit *circ, e_role role, uint32_t bitlen,
		  ifstream &if_add,
		  ifstream &if_rand)
{
  /*
   * Variables for reading the added value and the random value
   */
  T add_val;
  T rand_val;
  if(role == SERVER) { if_rand >> rand_val; }
  if(role == CLIENT) { if_add >> add_val; }
  share *add_sh = role == SERVER ? circ->PutDummyINGate(bitlen) : circ->PutINGate(add_val, bitlen, CLIENT);
  share *rand_sh = role == SERVER ? circ->PutINGate(rand_val, bitlen, SERVER) : circ->PutDummyINGate(bitlen);
  return circ->PutSUBGate(add_sh, rand_sh);
}

#endif
