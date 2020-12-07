/*

Authors: Mayank Rathee, Nishant Kumar.

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
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include "secondary.h"
#include "connect.h"
#include "AESObject.h"
#include "Functionalities.h"
using namespace std;

enum e_role { CLIENT, SERVER, ALL };

/*
 * we maintain a queue of outputs
 * basically every OUTPUT adds an OUTGate,
 * and adds the returned share to this queue
 * this queue is then flushed at the end after we have done exec
 */
struct output_queue_elmt {
  porthosLongUnsignedInt val;
  e_role role;  //who should we output the clear value to
  ostream& os;  //output stream to which we will output (cout or files), can this be a reference to prevent copying?
};

typedef  vector<output_queue_elmt> output_queue;
/*
 * called from the EzPC generated code
 */
void add_to_output_queue(output_queue &q,
			 porthosLongUnsignedInt val,
			 e_role role,
			 ostream &os)
{
  struct output_queue_elmt elmt { val, role, os };
  q.push_back(elmt);
}
/*
 * flush the queue
 * both parties call this function with their role
 * called from the EzPC generated code
 */
void flush_output_queue(output_queue &q, 
		e_role role)
{
  for(output_queue::iterator it = q.begin(); it != q.end(); ++it) {  //iterate over the queue
    if(it->role == ALL || it->role == role) {  //if the queue element role is same as mine
      it->os << static_cast<porthosLongSignedInt>(it->val) << endl;
    }
  }
}

#endif
