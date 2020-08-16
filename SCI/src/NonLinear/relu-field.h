/*
Authors: Mayank Rathee
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

#ifndef RELU_FIELD_H__
#define RELU_FIELD_H__

#include "globals.h"
#include "Millionaire/millionaire.h"
#include "NonLinear/drelu-field.h"

#define FIELD 1

template<typename IO, typename type>
class ReLUFieldProtocol : public ReLUProtocol<IO, type> 
{	
public:
	IO* io = nullptr;
	sci::OTPack<IO>* otpack = nullptr;
	TripleGenerator<IO>* triple_gen = nullptr;
	DReLUFieldProtocol<IO>* relu_triple_compare_oracle = nullptr;
	int party;
	int algeb_str;
	int l, b;
	int num_cmps;
	uint64_t p;
	uint64_t p_bitlen;
	uint64_t p_bitlen_triple_comparison;
	uint64_t rhs_wrap;
	uint64_t rhs_wrap_off;
	uint64_t rhs_wrap_on;
	uint8_t zero_small = 0;
	
	//Constructor
	ReLUFieldProtocol(
			int party, 
			int algeb_str, 
			IO* io, 
			int l, 
			int b, 
			uint64_t mod,
      sci::OTPack<IO>* otpack)
	{
		this->party = party;
		this->algeb_str = algeb_str;
		this->io = io;
		this->l = l;
		this->b = b;
		this->p = mod;
		this->p_bitlen = l;
		this->p_bitlen_triple_comparison = l+1;
		this->otpack = otpack;
		this->triple_gen = new TripleGenerator<IO>(party, io, otpack);
		this->relu_triple_compare_oracle = new DReLUFieldProtocol<IO>(party, l+1, b, mod, io, otpack);
		configure();
	}

	//Destructor
	~ReLUFieldProtocol()
	{
		delete triple_gen;
		delete relu_triple_compare_oracle;
	}

	void configure()
	{
		// Set all the inequality RHS values here.
		rhs_wrap = this->p;
		rhs_wrap_off = this->p/2;
		rhs_wrap_on = this->p + this->p/2;
	}

	//Tester Ideal Functionality
	void drelu_field_ideal_func(
			uint8_t* result, 
			uint64_t* sh1, 
			uint64_t* sh2, 
			int num_relu)
	{
		uint8_t *wrap, *relu_wrap_off, *relu_wrap_on, *expected_drelu, *actual_drelu;
		wrap = new uint8_t[num_relu];
		relu_wrap_off = new uint8_t[num_relu];
		relu_wrap_on = new uint8_t[num_relu];
		expected_drelu = new uint8_t[num_relu];
		actual_drelu = new uint8_t[num_relu];
		for(int i=0; i<num_relu; i++){
			wrap[i] = greater_than_wrap(sh1[i], sci::neg_mod((int64_t)(this->rhs_wrap - sh2[i]), this->p));
			relu_wrap_off[i] = greater_than_relu_wrap_off(sh1[i], sci::neg_mod((int64_t)(this->rhs_wrap_off - sh2[i]), this->p));
			relu_wrap_on[i] = greater_than_relu_wrap_on(sh1[i], sci::neg_mod((int64_t)(this->rhs_wrap_on - sh2[i]), this->p));
			expected_drelu[i] = (relu_wrap_on[i] & wrap[i]) ^ ((1 ^ wrap[i]) & relu_wrap_off[i]);
			expected_drelu[i] = 1 ^ expected_drelu[i];
			actual_drelu[i] = (((sh1[i] + sh2[i]) % this->p) > this->rhs_wrap_off);
			result[i] = 1 ^ actual_drelu[i];
			assert(((result[i] & 1) == (1 & expected_drelu[i])) && "The computed DReLU did not match the actual DReLU");
		}
	}

	// To handle the case when RHS is taken modulo p
	uint8_t greater_than_wrap(
			uint64_t lhs, 
			uint64_t rhs)
	{
		if(rhs == 0ULL){
			return 0;
		}
		else{
			return (lhs >= rhs);
		}
	}

	uint8_t greater_than_relu_wrap_off(
			uint64_t lhs, 
			uint64_t rhs)
	{
		if(rhs <= this->rhs_wrap_off){
			return (lhs > rhs);
		}
		else{
			return 1;
		}
	}

	uint8_t greater_than_relu_wrap_on(
			uint64_t lhs, 
			uint64_t rhs)
	{
		if(rhs <= this->rhs_wrap_off){
			return 0;
		}
		else{
			return (lhs > rhs);
		}
	}

	void drelu(
			uint8_t* drelu_res, 
			uint64_t* share, 
			int num_relu)
	{
			this->relu_triple_compare_oracle->compute_drelu(drelu_res, share, num_relu);
	}

	void relu(
			type* result, 
			type* share, 
			int num_relu,
      uint8_t* drelu_res = nullptr, 
			bool skip_ot = false)
	{
		uint8_t* drelu_ans = new uint8_t[num_relu];
		drelu(drelu_ans, (uint64_t*)share, num_relu);	
		if(drelu_res != nullptr){
			memcpy(drelu_res, drelu_ans, num_relu*sizeof(uint8_t));
		}
		if(skip_ot){
			delete[] drelu_ans;
			// std::cout<<"Doing Max and ArgMax OTs together"<<std::endl;
			return;
		}	
		// Now perform x.msb(x)
		// 2 OTs required with reversed roles
		uint64_t** ot_messages = new uint64_t*[num_relu];
		for(int i=0; i<num_relu; i++){
			ot_messages[i] = new uint64_t[2];
		}
		uint64_t* additive_masks = new uint64_t[num_relu];
		uint64_t* received_shares = new uint64_t[num_relu];
		this->triple_gen->prg->template random_mod_p<type>((type*)additive_masks, num_relu, this->p);
		for(int i=0; i<num_relu; i++){
			additive_masks[i] %= this->p;
		}
		
		switch (this->party) {
			case sci::ALICE: {
				for(int i=0; i<num_relu; i++){
					set_relu_end_ot_messages(ot_messages[i], share+i, drelu_ans+i, ((type*)additive_masks)+i);
				}
				otpack->iknp_straight->send(ot_messages, num_relu, this->l);
				otpack->iknp_reversed->recv(received_shares, drelu_ans, num_relu, this->l);	
				break;		
			}
			case sci::BOB: {
				for(int i=0; i<num_relu; i++){
					set_relu_end_ot_messages(ot_messages[i], share+i, drelu_ans+i, ((type*)additive_masks)+i);
				}
				otpack->iknp_straight->recv(received_shares, drelu_ans, num_relu, this->l);	
				otpack->iknp_reversed->send(ot_messages, num_relu, this->l);
				break;
			} 
		}
		for(int i=0; i<num_relu; i++){
			result[i] = ((type*)additive_masks)[i] + ((type*)received_shares)[(8/sizeof(type))*i];
			result[i] %= this->p;
		}
		delete[] additive_masks;
		delete[] received_shares;
		for(int i=0; i<num_relu; i++){
			delete[] ot_messages[i];
		}
		delete[] ot_messages;
	}

	void set_relu_end_ot_messages(
			uint64_t* ot_messages, 
			type* value_share, 
			uint8_t* xor_share, 
			type* additive_mask)
	{
		type temp0, temp1;
		temp0 = sci::neg_mod((int64_t)value_share[0] - (int64_t)additive_mask[0], this->p);
		temp1 = sci::neg_mod((int64_t)0LL - (int64_t)additive_mask[0], this->p);
		if(*xor_share == zero_small){
			ot_messages[0] = 0ULL + temp0;
			ot_messages[1] = 0ULL + temp1;
		}
		else{
			ot_messages[0] = 0ULL + temp1;
			ot_messages[1] = 0ULL + temp0;
		}
	}

};

#endif //RELU_FIELD_H__
