/*
Authors: Deevashwer Rathee
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

#define SCI_HE
#define BITLEN_41

#include "LinearHE/elemwise-prod-field.h"
#include "globals.h"

using namespace std;
using namespace seal;
using namespace sci;

int port = 8000;
string address;
bool localhost = true;
int vec_size = 56*56;
int filter_precision = 15;

void ElemWiseProd(
        ElemWiseProdField &he_prod,
        int32_t size)
{
    vector<uint64_t> inArr(size);
    vector<uint64_t> multArr(size);
    vector<uint64_t> outArr(size);
    PRG128 prg;
    if (party == SERVER) {
        prg.random_data(multArr.data(), size*sizeof(uint64_t));
        for(int i = 0; i < size; i++) {
            multArr[i] = ((int64_t) multArr[i]) >> (64 - filter_precision);
        }
    }
    prg.random_mod_p<uint64_t>(inArr.data(), size, prime_mod);

    INIT_TIMER;
    START_TIMER;
    he_prod.elemwise_product(size, inArr, multArr, outArr, true, true);
	STOP_TIMER("Total Time for ElemWiseProduct");
}

int main(int argc, char** argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("s", vec_size, "Size of Vectors");
    amap.arg("fp", filter_precision, "Filter Precision");
    amap.arg("lo", localhost, "Localhost Run?");
    amap.parse(argc, argv);

    if(not localhost) {
#if defined(LAN_EXEC)
        address = "40.118.124.169"; // SGX4
#elif defined(WAN_EXEC)
        address = "40.117.40.111"; // SGX2
#endif
    } else {
        address = "127.0.0.1";
    }

    cout << "========================================================================" << endl;
    cout << "Role: " << party << " - Bitlength: " << bitlength << " - Vector Size: "
        << vec_size << " - # Threads: " << numThreads << endl;
    cout << "========================================================================" << endl;

    io = new NetIO(party==1 ? nullptr:address.c_str(), port);

    ElemWiseProdField he_prod(party, io);

    ElemWiseProd(he_prod, vec_size);
    return 0;
}
