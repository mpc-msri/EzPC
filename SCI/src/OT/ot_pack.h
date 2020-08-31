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

#ifndef OT_PACK_H__
#define OT_PACK_H__
#include "utils/emp-tool.h"
#include "OT/emp-ot.h"

/*
 * DISCLAIMER:
 * OTPack avoids computing PRG keys for each OT instance by reusing the keys generated
 * (through base OTs) for another OT instance. Ideally, the PRGs within OT instances,
 * using the same keys, should use mutually exclusive counters for security. However,
 * the current implementation does not support this.
 */

namespace sci {
template<typename T>
class OTPack {
public:
	SplitKKOT<T> *kkot_beta;
	SplitKKOT<T> *kkot_4;
	SplitKKOT<T> *kkot_16;
	SplitKKOT<T> *kkot_8;
	// iknp_straight and iknp_reversed: party
	// acts as sender in straight and receiver in reversed.
	// Needed for MUX calls.
	SplitIKNP<T> *iknp_straight;
	SplitIKNP<T> *iknp_reversed;
	T *io;
	int beta, r, l, party;
	bool do_setup = false;

	OTPack(T* io, int party, int beta, int l, bool do_setup = true){
		this->beta = beta;
		this->l = l;
		this->r = l%beta;
		this->party = party;
		this->do_setup = do_setup;
		this->io = io;

		kkot_beta = new SplitKKOT<NetIO>(party, io, 1<<beta);
		kkot_4 = new SplitKKOT<NetIO>(party, io, 4);
        kkot_16 = new SplitKKOT<NetIO>(party, io, 16);
        kkot_8 = new SplitKKOT<NetIO>(party, io, 8);

        iknp_straight = new SplitIKNP<NetIO>(party, io);
        iknp_reversed = new SplitIKNP<NetIO>(3-party, io);

		if(do_setup){
			SetupBaseOTs();
		}
	}
	
	~OTPack(){
		delete kkot_beta;
		delete kkot_4;
		delete kkot_8;
		delete kkot_16;
		delete iknp_straight;
		delete iknp_reversed;
	}

	void SetupBaseOTs(){
    switch (party) {
			case 1:
				kkot_beta->setup_send();
				iknp_straight->setup_send();
				iknp_reversed->setup_recv();
				kkot_4->setup_send(kkot_beta->k0, kkot_beta->s);
				kkot_16->setup_send(kkot_beta->k0, kkot_beta->s);
				kkot_8->setup_send(kkot_beta->k0, kkot_beta->s);
				break;
			case 2:
				kkot_beta->setup_recv();
				iknp_straight->setup_recv();
				iknp_reversed->setup_send();
				kkot_4->setup_recv(kkot_beta->k0, kkot_beta->k1);
				kkot_16->setup_recv(kkot_beta->k0, kkot_beta->k1);
				kkot_8->setup_recv(kkot_beta->k0, kkot_beta->k1);
				break;
    }
	}

	OTPack<T>* operator=(OTPack<T> *copy_from){
		assert(this->do_setup == false && copy_from->do_setup == true);
		OTPack<T> *temp = new OTPack<T>(this->io, copy_from->party, copy_from->beta, copy_from->l, false);
		SplitKKOT<T> *kkot_base = copy_from->kkot_beta;
		SplitIKNP<T> *iknp_s_base = copy_from->iknp_straight;
		SplitIKNP<T> *iknp_r_base = copy_from->iknp_reversed;

        switch (party) {
			case 1:
				temp->kkot_beta->setup_send(kkot_base->k0, kkot_base->s);
				temp->iknp_straight->setup_send(iknp_s_base->k0, iknp_s_base->s);
				temp->iknp_reversed->setup_recv(iknp_r_base->k0, iknp_r_base->k1);
				temp->kkot_4->setup_send(temp->kkot_beta->k0, temp->kkot_beta->s);
				temp->kkot_16->setup_send(temp->kkot_beta->k0, temp->kkot_beta->s);
				temp->kkot_8->setup_send(temp->kkot_beta->k0, temp->kkot_beta->s);
				break;
			case 2:
				temp->kkot_beta->setup_recv(kkot_base->k0, kkot_base->k1);
				temp->iknp_straight->setup_recv(iknp_s_base->k0, iknp_s_base->k1);
				temp->iknp_reversed->setup_send(iknp_s_base->k0, iknp_s_base->s);
				temp->kkot_4->setup_recv(temp->kkot_beta->k0, temp->kkot_beta->k1);
				temp->kkot_16->setup_recv(temp->kkot_beta->k0, temp->kkot_beta->k1);
				temp->kkot_8->setup_recv(temp->kkot_beta->k0, temp->kkot_beta->k1);
				break;
        }
        temp->do_setup = true;
        return temp;
    }

	void copy(OTPack<T> *copy_from){
		assert(this->do_setup == false && copy_from->do_setup == true);
		SplitKKOT<T> *kkot_base = copy_from->kkot_beta;
		SplitIKNP<T> *iknp_s_base = copy_from->iknp_straight;
		SplitIKNP<T> *iknp_r_base = copy_from->iknp_reversed;

        switch (this->party) {
			case 1:
				this->kkot_beta->setup_send(kkot_base->k0, kkot_base->s);
				this->iknp_straight->setup_send(iknp_s_base->k0, iknp_s_base->s);
				this->iknp_reversed->setup_recv(iknp_r_base->k0, iknp_r_base->k1);
				this->kkot_4->setup_send(this->kkot_beta->k0, this->kkot_beta->s);
				this->kkot_16->setup_send(this->kkot_beta->k0, this->kkot_beta->s);
				this->kkot_8->setup_send(this->kkot_beta->k0, this->kkot_beta->s);
				break;
			case 2:
				this->kkot_beta->setup_recv(kkot_base->k0, kkot_base->k1);
				this->iknp_straight->setup_recv(iknp_s_base->k0, iknp_s_base->k1);
				this->iknp_reversed->setup_send(iknp_r_base->k0, iknp_r_base->s);
				this->kkot_4->setup_recv(this->kkot_beta->k0, this->kkot_beta->k1);
				this->kkot_16->setup_recv(this->kkot_beta->k0, this->kkot_beta->k1);
				this->kkot_8->setup_recv(this->kkot_beta->k0, this->kkot_beta->k1);
				break;
        }
        this->do_setup = true;
        return;
    }

};
}
#endif // OT_PACK_H__
