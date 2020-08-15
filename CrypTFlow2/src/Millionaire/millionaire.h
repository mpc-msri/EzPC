/*
Authors: Deevashwer Rathee, Mayank Rathee
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

#ifndef MILLIONAIRE_H__
#define MILLIONAIRE_H__
#include "OT/emp-ot.h"
#include "utils/emp-tool.h"
#include "Millionaire/bit-triple-generator.h"
#include <cmath>

template<typename IO>
class MillionaireProtocol {
	public:
		IO* io = nullptr;
		sci::OTPack<IO>* otpack;
		TripleGenerator<IO>* triple_gen;
		int party;
		int l, r, log_alpha, beta, beta_pow;
		int num_digits, num_triples_corr, num_triples_std, log_num_digits;
		int num_triples;
		uint8_t mask_beta, mask_r;

		MillionaireProtocol(int party,
				int bitlength,
				int log_radix_base,
				IO* io,
				sci::OTPack<IO> *otpack)
		{
			assert(log_radix_base <= 8);
			assert(bitlength <= 64);
			this->party = party;
			this->l = bitlength;
			this->beta = log_radix_base;
			this->io = io;
			this->otpack = otpack;
			this->triple_gen = new TripleGenerator<IO>(party, io, otpack);
			configure();
		}

		void configure()
		{
			this->num_digits = ceil((double)l/beta);
			this->r = l % beta;
			this->log_alpha = sci::bitlen(num_digits) - 1;
			this->log_num_digits = log_alpha + 1;
			this->num_triples_corr = 2*num_digits - 2 - 2*log_num_digits;
			this->num_triples_std = log_num_digits;
			this->num_triples = num_triples_std + num_triples_corr;
			if (beta == 8) this->mask_beta = -1;
			else this->mask_beta = (1 << beta) - 1;
			this->mask_r = (1 << r) - 1;
			this->beta_pow = 1 << beta;
		}

		~MillionaireProtocol()
		{
			delete triple_gen;
		}

		void compare(uint8_t* res,
				uint64_t* data,
				int num_cmps,
				bool greater_than = true)
		{
			uint8_t* digits; // num_digits * num_cmps
			uint8_t* leaf_res_cmp; // num_digits * num_cmps
			uint8_t* leaf_res_eq; // num_digits * num_cmps

			digits = new uint8_t[num_digits*num_cmps];
			leaf_res_cmp = new uint8_t[num_digits*num_cmps];
			leaf_res_eq = new uint8_t[num_digits*num_cmps];

			// Extract radix-digits from data
			for(int i = 0; i < num_digits; i++) // Stored from LSB to MSB
				for(int j = 0; j < num_cmps; j++)
					if ((i == num_digits-1) && (r != 0))
						digits[i*num_cmps+j] = (uint8_t)(data[j] >> i*beta) & mask_r;
					else
						digits[i*num_cmps+j] = (uint8_t)(data[j] >> i*beta) & mask_beta;

			if(party == sci::ALICE)
			{
				uint8_t** leaf_ot_messages; // (num_digits * num_cmps) X beta_pow (=2^beta)
				leaf_ot_messages = new uint8_t*[num_digits*num_cmps];
				for(int i = 0; i < num_digits*num_cmps; i++)
					leaf_ot_messages[i] = new uint8_t[beta_pow];

				// Set Leaf OT messages
				triple_gen->prg->random_bool((bool*)leaf_res_cmp, num_digits*num_cmps);
				triple_gen->prg->random_bool((bool*)leaf_res_eq, num_digits*num_cmps);
				for(int i = 0; i < num_digits; i++) {
					for(int j = 0; j < num_cmps; j++) {
						if (i == 0){
							set_leaf_ot_messages(leaf_ot_messages[i*num_cmps+j], digits[i*num_cmps+j],
									beta_pow, leaf_res_cmp[i*num_cmps+j], 0, greater_than, false);
						}
						else if (i == (num_digits - 1) && (r > 0)){
#ifdef WAN_EXEC
							set_leaf_ot_messages(leaf_ot_messages[i*num_cmps+j], digits[i*num_cmps+j],
									beta_pow, leaf_res_cmp[i*num_cmps+j],
									leaf_res_eq[i*num_cmps+j], greater_than);
#else
							set_leaf_ot_messages(leaf_ot_messages[i*num_cmps+j], digits[i*num_cmps+j],
									1 << r, leaf_res_cmp[i*num_cmps+j],
									leaf_res_eq[i*num_cmps+j], greater_than);
#endif
						}
						else{
							set_leaf_ot_messages(leaf_ot_messages[i*num_cmps+j], digits[i*num_cmps+j],
									beta_pow, leaf_res_cmp[i*num_cmps+j],
									leaf_res_eq[i*num_cmps+j], greater_than);
						}
					}
				}

				// Perform Leaf OTs
#ifdef WAN_EXEC
				otpack->kkot_beta->send(leaf_ot_messages, num_cmps*(num_digits), 2);
#else
				otpack->kkot_beta->send(leaf_ot_messages, num_cmps, 1);
				if (r == 1) {
					otpack->kkot_beta->send(leaf_ot_messages+num_cmps, num_cmps*(num_digits-2), 2);
					otpack->iknp_straight->send(leaf_ot_messages+num_cmps*(num_digits-1), num_cmps, 2);
				}
				else if (r != 0) {
					otpack->kkot_beta->send(leaf_ot_messages+num_cmps, num_cmps*(num_digits-2), 2);
					if(r == 2){
						otpack->kkot_4->send(leaf_ot_messages+num_cmps*(num_digits-1), num_cmps, 2);
					}
					else if(r == 3){
						otpack->kkot_8->send(leaf_ot_messages+num_cmps*(num_digits-1), num_cmps, 2);
					}
					else if(r == 4){
						otpack->kkot_16->send(leaf_ot_messages+num_cmps*(num_digits-1), num_cmps, 2);
					}
					else{
						throw std::invalid_argument("Not yet implemented!");
					}
				}
				else {
					otpack->kkot_beta->send(leaf_ot_messages+num_cmps, num_cmps*(num_digits-1), 2);
				}
#endif
				// Cleanup
				for(int i = 0; i < num_digits*num_cmps; i++)
					delete[] leaf_ot_messages[i];
				delete[] leaf_ot_messages;
			}
			else // party = sci::BOB
			{
				// Perform Leaf OTs
#ifdef WAN_EXEC
				otpack->kkot_beta->recv(leaf_res_cmp, digits, num_cmps*(num_digits), 2);
#else
				otpack->kkot_beta->recv(leaf_res_cmp, digits, num_cmps, 1);
				if (r == 1) {
					otpack->kkot_beta->recv(leaf_res_cmp+num_cmps, digits+num_cmps, num_cmps*(num_digits-2), 2);
					otpack->iknp_straight->recv(leaf_res_cmp+num_cmps*(num_digits-1),
							digits+num_cmps*(num_digits-1), num_cmps, 2);
				}
				else if (r != 0) {
					otpack->kkot_beta->recv(leaf_res_cmp+num_cmps, digits+num_cmps, num_cmps*(num_digits-2), 2);
					if(r == 2){
						otpack->kkot_4->recv(leaf_res_cmp+num_cmps*(num_digits-1),
								digits+num_cmps*(num_digits-1), num_cmps, 2);
					}
					else if(r == 3){
						otpack->kkot_8->recv(leaf_res_cmp+num_cmps*(num_digits-1),
								digits+num_cmps*(num_digits-1), num_cmps, 2);
					}
					else if(r == 4){
						otpack->kkot_16->recv(leaf_res_cmp+num_cmps*(num_digits-1),
								digits+num_cmps*(num_digits-1), num_cmps, 2);
					}
					else{
						throw std::invalid_argument("Not yet implemented!");
					}
				}
				else {
					otpack->kkot_beta->recv(leaf_res_cmp+num_cmps, digits+num_cmps, num_cmps*(num_digits-1), 2);
				}
#endif

				// Extract equality result from leaf_res_cmp
				for(int i = num_cmps; i < num_digits*num_cmps; i++) {
					leaf_res_eq[i] = leaf_res_cmp[i] & 1;
					leaf_res_cmp[i] >>= 1;
				}
			}

			traverse_and_compute_ANDs(num_cmps, leaf_res_eq, leaf_res_cmp);

			for (int i = 0; i < num_cmps; i++)
				res[i] = leaf_res_cmp[i];

			// Cleanup
			delete[] digits;
			delete[] leaf_res_cmp;
			delete[] leaf_res_eq;
		}

		void set_leaf_ot_messages(uint8_t* ot_messages,
				uint8_t digit,
				int N,
				uint8_t mask_cmp,
				uint8_t mask_eq,
				bool greater_than,
				bool eq = true)
		{
			for(int i = 0; i < N; i++) {
				if (greater_than) {
					ot_messages[i] = ((digit > i) ^ mask_cmp);
				} else {
					ot_messages[i] = ((digit < i) ^ mask_cmp);
				}
				if (eq) {
					ot_messages[i] = (ot_messages[i] << 1) | ((digit == i) ^ mask_eq);
				}
			}
		}

		/**************************************************************************************************
		 *                         AND computation related functions
		 **************************************************************************************************/

		void traverse_and_compute_ANDs(int num_cmps, uint8_t* leaf_res_eq, uint8_t* leaf_res_cmp){
#ifdef WAN_EXEC
			Triple triples_std((num_triples)*num_cmps, true);
#else
			Triple triples_corr(num_triples_corr*num_cmps, true, num_cmps);
			Triple triples_std(num_triples_std*num_cmps, true);
#endif
			// Generate required Bit-Triples
#ifdef WAN_EXEC
			//std::cout<<"Running on WAN_EXEC; Skipping correlated triples"<<std::endl;
			triple_gen->generate(party, &triples_std, _16KKOT_to_4OT);
#else
			triple_gen->generate(party, &triples_corr, _8KKOT);
			triple_gen->generate(party, &triples_std, _16KKOT_to_4OT);
#endif
			// std::cout << "Bit Triples Generated" << std::endl;

			// Combine leaf OT results in a bottom-up fashion
			int counter_std = 0, old_counter_std = 0;
			int counter_corr = 0, old_counter_corr = 0;
			int counter_combined = 0, old_counter_combined = 0;
			uint8_t* ei = new uint8_t[(num_triples*num_cmps)/8];
			uint8_t* fi = new uint8_t[(num_triples*num_cmps)/8];
			uint8_t* e = new uint8_t[(num_triples*num_cmps)/8];
			uint8_t* f = new uint8_t[(num_triples*num_cmps)/8];

			for(int i = 1; i < num_digits; i*=2) {
				for(int j = 0; j < num_digits and j+i < num_digits; j += 2*i) {
					if (j == 0) {
#ifdef WAN_EXEC
						AND_step_1(ei+(counter_std*num_cmps)/8, fi+(counter_std*num_cmps)/8,
								leaf_res_cmp+j*num_cmps, leaf_res_eq+(j+i)*num_cmps,
								(triples_std.ai)+(counter_combined*num_cmps)/8,
								(triples_std.bi)+(counter_combined*num_cmps)/8, num_cmps);
						counter_std++;
						counter_combined++;
#else
						AND_step_1(ei+(counter_std*num_cmps)/8, fi+(counter_std*num_cmps)/8,
								leaf_res_cmp+j*num_cmps, leaf_res_eq+(j+i)*num_cmps,
								(triples_std.ai)+(counter_std*num_cmps)/8,
								(triples_std.bi)+(counter_std*num_cmps)/8, num_cmps);
						counter_std++;
#endif
					} else {
#ifdef WAN_EXEC
						AND_step_1(ei+((num_triples_std+2*counter_corr)*num_cmps)/8,
								fi+((num_triples_std+2*counter_corr)*num_cmps)/8,
								leaf_res_cmp+j*num_cmps, leaf_res_eq+(j+i)*num_cmps,
								(triples_std.ai)+(counter_combined*num_cmps)/8,
								(triples_std.bi)+(counter_combined*num_cmps)/8, num_cmps);
						counter_combined++;
						AND_step_1(ei+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								fi+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								leaf_res_eq+j*num_cmps, leaf_res_eq+(j+i)*num_cmps,
								(triples_std.ai)+(counter_combined*num_cmps)/8,
								(triples_std.bi)+(counter_combined*num_cmps)/8, num_cmps);
						counter_combined++;
						counter_corr++;
#else
						AND_step_1(ei+((num_triples_std+2*counter_corr)*num_cmps)/8,
								fi+((num_triples_std+2*counter_corr)*num_cmps)/8,
								leaf_res_cmp+j*num_cmps, leaf_res_eq+(j+i)*num_cmps,
								(triples_corr.ai)+(2*counter_corr*num_cmps)/8,
								(triples_corr.bi)+(2*counter_corr*num_cmps)/8, num_cmps);
						AND_step_1(ei+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								fi+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								leaf_res_eq+j*num_cmps, leaf_res_eq+(j+i)*num_cmps,
								(triples_corr.ai)+((2*counter_corr+1)*num_cmps)/8,
								(triples_corr.bi)+((2*counter_corr+1)*num_cmps)/8, num_cmps);
						counter_corr++;
#endif
					}
				}
				int offset_std = (old_counter_std*num_cmps)/8;
				int size_std = ((counter_std - old_counter_std)*num_cmps)/8;
				int offset_corr = ((num_triples_std + 2*old_counter_corr)*num_cmps)/8;
				int size_corr = (2*(counter_corr - old_counter_corr)*num_cmps)/8;

				if(party == sci::ALICE)
				{
					io->send_data(ei+offset_std, size_std);
					io->send_data(ei+offset_corr, size_corr);
					io->send_data(fi+offset_std, size_std);
					io->send_data(fi+offset_corr, size_corr);
					io->recv_data(e+offset_std, size_std);
					io->recv_data(e+offset_corr, size_corr);
					io->recv_data(f+offset_std, size_std);
					io->recv_data(f+offset_corr, size_corr);
				}
				else // party = sci::BOB
				{
					io->recv_data(e+offset_std, size_std);
					io->recv_data(e+offset_corr, size_corr);
					io->recv_data(f+offset_std, size_std);
					io->recv_data(f+offset_corr, size_corr);
					io->send_data(ei+offset_std, size_std);
					io->send_data(ei+offset_corr, size_corr);
					io->send_data(fi+offset_std, size_std);
					io->send_data(fi+offset_corr, size_corr);
				}
				for(int i = 0; i < size_std; i++) {
					e[i+offset_std] ^= ei[i+offset_std];
					f[i+offset_std] ^= fi[i+offset_std];
				}
				for(int i = 0; i < size_corr; i++) {
					e[i+offset_corr] ^= ei[i+offset_corr];
					f[i+offset_corr] ^= fi[i+offset_corr];
				}

				counter_std = old_counter_std;
				counter_corr = old_counter_corr;
#ifdef WAN_EXEC
				counter_combined = old_counter_combined;
#endif
				for(int j = 0; j < num_digits and j+i < num_digits; j += 2*i) {
					if (j == 0) {
#ifdef WAN_EXEC
						AND_step_2(leaf_res_cmp+j*num_cmps,
								e+(counter_std*num_cmps)/8,
								f+(counter_std*num_cmps)/8,
								ei+(counter_std*num_cmps)/8,
								fi+(counter_std*num_cmps)/8,
								(triples_std.ai)+(counter_combined*num_cmps)/8,
								(triples_std.bi)+(counter_combined*num_cmps)/8,
								(triples_std.ci)+(counter_combined*num_cmps)/8, num_cmps);
						counter_combined++;
#else
						AND_step_2(leaf_res_cmp+j*num_cmps,
								e+(counter_std*num_cmps)/8,
								f+(counter_std*num_cmps)/8,
								ei+(counter_std*num_cmps)/8,
								fi+(counter_std*num_cmps)/8,
								(triples_std.ai)+(counter_std*num_cmps)/8,
								(triples_std.bi)+(counter_std*num_cmps)/8,
								(triples_std.ci)+(counter_std*num_cmps)/8, num_cmps);
#endif
						for(int k = 0; k < num_cmps; k++)
							leaf_res_cmp[j*num_cmps+k] ^= leaf_res_cmp[(j+i)*num_cmps+k];
						counter_std++;
					} else {
#ifdef WAN_EXEC
						AND_step_2(leaf_res_cmp+j*num_cmps,
								e+((num_triples_std+2*counter_corr)*num_cmps)/8,
								f+((num_triples_std+2*counter_corr)*num_cmps)/8,
								ei+((num_triples_std+2*counter_corr)*num_cmps)/8,
								fi+((num_triples_std+2*counter_corr)*num_cmps)/8,
								(triples_std.ai)+(counter_combined*num_cmps)/8,
								(triples_std.bi)+(counter_combined*num_cmps)/8,
								(triples_std.ci)+(counter_combined*num_cmps)/8, num_cmps);
						counter_combined++;
						AND_step_2(leaf_res_eq+j*num_cmps,
								e+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								f+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								ei+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								fi+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								(triples_std.ai)+(counter_combined*num_cmps)/8,
								(triples_std.bi)+(counter_combined*num_cmps)/8,
								(triples_std.ci)+(counter_combined*num_cmps)/8, num_cmps);
						counter_combined++;
#else
						AND_step_2(leaf_res_cmp+j*num_cmps,
								e+((num_triples_std+2*counter_corr)*num_cmps)/8,
								f+((num_triples_std+2*counter_corr)*num_cmps)/8,
								ei+((num_triples_std+2*counter_corr)*num_cmps)/8,
								fi+((num_triples_std+2*counter_corr)*num_cmps)/8,
								(triples_corr.ai)+(2*counter_corr*num_cmps)/8,
								(triples_corr.bi)+(2*counter_corr*num_cmps)/8,
								(triples_corr.ci)+(2*counter_corr*num_cmps)/8, num_cmps);
						AND_step_2(leaf_res_eq+j*num_cmps,
								e+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								f+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								ei+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								fi+((num_triples_std+(2*counter_corr+1))*num_cmps)/8,
								(triples_corr.ai)+((2*counter_corr+1)*num_cmps)/8,
								(triples_corr.bi)+((2*counter_corr+1)*num_cmps)/8,
								(triples_corr.ci)+((2*counter_corr+1)*num_cmps)/8, num_cmps);
#endif
						for(int k = 0; k < num_cmps; k++)
							leaf_res_cmp[j*num_cmps+k] ^= leaf_res_cmp[(j+i)*num_cmps+k];
						counter_corr++;
					}
				}
				old_counter_std = counter_std;
				old_counter_corr = counter_corr;
#ifdef WAN_EXEC
				old_counter_combined = counter_combined;
#endif
			}

#ifdef WAN_EXEC
			assert(counter_combined == num_triples);
#else
			assert(counter_std == num_triples_std);
			assert(2*counter_corr == num_triples_corr);
#endif

			//cleanup
			delete[] ei;
			delete[] fi;
			delete[] e;
			delete[] f;

		}

		void AND_step_1(uint8_t* ei, // evaluates batch of 8 ANDs
				uint8_t* fi,
				uint8_t* xi,
				uint8_t* yi,
				uint8_t* ai,
				uint8_t* bi,
				int num_ANDs) {
			assert(num_ANDs % 8 == 0);
			for(int i = 0; i < num_ANDs; i+=8) {
				ei[i/8] = ai[i/8];
				fi[i/8] = bi[i/8];
				ei[i/8] ^= sci::bool_to_uint8(xi+i, 8);
				fi[i/8] ^= sci::bool_to_uint8(yi+i, 8);
			}
		}
		void AND_step_2(uint8_t* zi, // evaluates batch of 8 ANDs
				uint8_t* e,
				uint8_t* f,
				uint8_t* ei,
				uint8_t* fi,
				uint8_t* ai,
				uint8_t* bi,
				uint8_t* ci,
				int num_ANDs)
		{
			assert(num_ANDs % 8 == 0);
			for(int i = 0; i < num_ANDs; i+=8) {
				uint8_t temp_z;
				if (party == sci::ALICE)
					temp_z = e[i/8] & f[i/8];
				else
					temp_z = 0;
				temp_z ^= f[i/8] & ai[i/8];
				temp_z ^= e[i/8] & bi[i/8];
				temp_z ^= ci[i/8];
				sci::uint8_to_bool(zi+i, temp_z, 8);
			}
		}
};

#endif //MILLIONAIRE_H__
