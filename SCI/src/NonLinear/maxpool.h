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

#ifndef MAXPOOL_PRIMARY_H__
#define MAXPOOL_PRIMARY_H__

#include "NonLinear/relu-ring.h"
#include "NonLinear/relu-field.h"

template<typename IO, typename type>
class MaxPoolProtocol 
{
public:
	IO* io = nullptr;
	sci::OTPack<IO>* otpack = nullptr;
	TripleGenerator<IO>* triple_gen = nullptr;
	ReLURingProtocol<IO, type>* relu_oracle = nullptr;
	ReLUFieldProtocol<IO, type>* relu_field_oracle = nullptr;
	int party;
	int algeb_str;
	int l, b;
	int num_cmps;
	uint64_t prime_mod;
	type mask_l;

	//Constructor
	MaxPoolProtocol(
			int party, 
			int algeb_str, 
			IO* io, 
			int l, 
			int b, 
			uint64_t prime,
      sci::OTPack<IO> *otpack, 
			ReLUProtocol<IO, type>* relu_obj = nullptr)
	{
		this->party = party;
		this->algeb_str = algeb_str;
		this->io = io;
		this->l = l;
		this->b = b;
    this->prime_mod = prime;
    this->otpack = otpack;
		if(algeb_str == RING){
			if(relu_obj == nullptr){
                this->relu_oracle = new ReLURingProtocol<IO, type>(party, RING, io, l, b, otpack);
			} else {
                this->relu_oracle = (ReLURingProtocol<IO, type>*) relu_obj;
			}
		}
		else{
			if(relu_obj == nullptr){
                this->relu_field_oracle = new ReLUFieldProtocol<IO, type>(party, FIELD, io, l, b, this->prime_mod, otpack);
			}
			else{
                this->relu_field_oracle = (ReLUFieldProtocol<IO, type>*) relu_obj;
			}
		}
		configure();
	}

	//Destructor
	~MaxPoolProtocol()
	{
		//Empty
	}

	void configure()
	{
		if(this->l != 32 && this->l != 64){
			mask_l = (type)((1ULL<<l)-1);
		}
		else if(this->l == 32){
			mask_l = -1;
		}
		else{ //l = 64
			mask_l = -1ULL;
		}

	}

	void funcMaxMPC(
			int rows, 
			int cols, 
			type* inpArr, 
			type* maxi, 
			type* maxiIdx, 
			bool computeMaxIdx=false)
	{
		type* max_temp = new type[rows];
		type* compare_with = new type[rows];
		if(this->algeb_str == FIELD){
			for(int r=0; r<rows; r++){
				max_temp[r] = inpArr[r*cols];
			}
			for(int c=1; c<cols; c++){
				for(int r=0; r<rows; r++){
					compare_with[r] = sci::neg_mod((int64_t)((int64_t)max_temp[r] - (int64_t)inpArr[r*cols + c]), this->prime_mod);
				}
				relu_field_oracle->relu(max_temp, compare_with, rows);
				for(int r=0; r<rows; r++){
					max_temp[r] = (max_temp[r] + inpArr[r*cols + c]) % this->prime_mod;
				}
			}
			for(int r=0; r<rows; r++){
				maxi[r] = max_temp[r];
			}
		}
		else{ //RING
			for(int r=0; r<rows; r++){
				max_temp[r] = inpArr[r*cols];
			}
			for(int c=1; c<cols; c++){
				for(int r=0; r<rows; r++){
						compare_with[r] = max_temp[r] - inpArr[r*cols + c];
				}
				relu_oracle->relu(max_temp, compare_with, rows);
				for(int r=0; r<rows; r++){
					max_temp[r] += inpArr[r*cols + c];
				}
			}
			for(int r=0; r<rows; r++){
				maxi[r] = max_temp[r];
				maxi[r] &= mask_l;
			}
		}	
	}

	void funcMaxMPCIdeal(
			int rows, 
			int cols, 
			type* inpArr, 
			type* maxi, 
			type* maxiIdx, 
			bool computeMaxIdx=false)
	{
		type* otherPartyData = new type[rows*cols];
		if (party==SERVER){
			io->send_data(inpArr, sizeof(type)*rows*cols);
			for(int i=0;i<rows;i++){
				maxi[i] = 0;
				maxiIdx[i] = 0;
			}
		}
		else{
			io->recv_data(otherPartyData, sizeof(type)*rows*cols);
			if (this->algeb_str==RING){
				for(int i=0;i<rows*cols;i++){
					otherPartyData[i] = otherPartyData[i] + inpArr[i];
				}
				for(int i=0;i<rows;i++){
					maxi[i] = otherPartyData[i*cols];
					maxiIdx[i] = 0;
				}
				for(int j=1;j<cols;j++){
					for(int i=0;i<rows;i++){
						if (((signedIntType)otherPartyData[i*cols+j])>((signedIntType)maxi[i])) {
							//Do signed comparison
							maxi[i] = otherPartyData[i*cols+j];
							maxiIdx[i] = j;
						}
					}
				}
			}
			else{
				for(int i=0;i<rows*cols;i++){
					otherPartyData[i] = (otherPartyData[i] + inpArr[i])%prime_mod;
				}
				for(int i=0;i<rows;i++){
					maxi[i] = otherPartyData[i*cols];
					maxiIdx[i] = 0;
				}
				for(int j=1;j<cols;j++){
					for(int i=0;i<rows;i++){
						int64_t curValSigned = otherPartyData[i*cols+j];
						int64_t curMaxSigned = maxi[i];
						if (curValSigned > (prime_mod/2)){
							curValSigned = curValSigned - prime_mod;
						}
						if (curMaxSigned > (prime_mod/2)){
							curMaxSigned = curMaxSigned - prime_mod;
						}
						if (curValSigned>curMaxSigned) {
							//Do signed comparison
							maxi[i] = otherPartyData[i*cols+j];
							maxiIdx[i] = j;
						}
					}
				}
			}
		}
		delete[] otherPartyData;
	}
};

#endif //MAXPOOL_PRIMARY_H__
