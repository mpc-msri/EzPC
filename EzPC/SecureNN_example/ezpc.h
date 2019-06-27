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
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
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
  uint64_t val;
  e_role role;  //who should we output the clear value to
  ostream& os;  //output stream to which we will output (cout or files), can this be a reference to prevent copying?
};
typedef  vector<output_queue_elmt> output_queue;
/*
 * called from the EzPC generated code
 */
void add_to_output_queue(output_queue &q,
			 uint64_t val,
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
void flush_output_queue(output_queue &q, e_role role)
{
  for(output_queue::iterator it = q.begin(); it != q.end(); ++it) {  //iterate over the queue
    if(it->role == ALL || it->role == role) {  //if the queue element role is same as mine
      it->os << static_cast<int64_t>(it->val) << endl;
    }
  }
}

#endif
