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

#include "LinearHE/conv-field.h"
#include "globals.h"

using namespace std;
using namespace seal;
using namespace sci;

int port = 8000;
string address;
bool localhost = true;
int image_h = 56;
int inp_chans = 64;
int filter_h = 3;
int out_chans = 64;
int pad_l = 0;
int pad_r = 0;
int stride = 2;
int filter_precision = 12;

void Conv(
        ConvField &he_conv,
        int32_t H,
        int32_t CI,
        int32_t FH,
        int32_t CO,
        int32_t zPadHLeft,
        int32_t zPadHRight,
        int32_t strideH)
{
    int newH = 1 + (H+zPadHLeft+zPadHRight-FH)/strideH;
    int N = 1;
    int W = H;
    int FW = FH;
    int zPadWLeft = zPadHLeft;
    int zPadWRight = zPadHRight;
    int strideW = strideH;
    int newW = newH;
    vector<vector<vector<vector<uint64_t>>>> inputArr(N);
    vector<vector<vector<vector<uint64_t>>>> filterArr(FH);
    vector<vector<vector<vector<uint64_t>>>> outArr(N);

    PRG128 prg;
    for(int i = 0; i < N; i++){
        outArr[i].resize(newH);
        for(int j = 0; j < newH; j++) {
            outArr[i][j].resize(newW);
            for(int k = 0; k < newW; k++) {
                outArr[i][j][k].resize(CO);
            }
        }
    }
    if(party == SERVER) {
        for(int i = 0; i < FH; i++){
            filterArr[i].resize(FW);
            for(int j = 0; j < FW; j++) {
                filterArr[i][j].resize(CI);
                for(int k = 0; k < CI; k++) {
                    filterArr[i][j][k].resize(CO);
                    prg.random_data(filterArr[i][j][k].data(), CO*sizeof(uint64_t));
                    for(int h = 0; h < CO; h++) {
                        filterArr[i][j][k][h]
                            = ((int64_t) filterArr[i][j][k][h]) >> (64 - filter_precision);
                    }
                }
            }
        }
    }
    for(int i = 0; i < N; i++){
        inputArr[i].resize(H);
        for(int j = 0; j < H; j++) {
            inputArr[i][j].resize(W);
            for(int k = 0; k < W; k++) {
                inputArr[i][j][k].resize(CI);
                prg.random_mod_p<uint64_t>(inputArr[i][j][k].data(), CI, prime_mod);
            }
        }
    }
    uint64_t comm_start = he_conv.io->counter;
    INIT_TIMER;
    START_TIMER;
    he_conv.convolution(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
            zPadWRight, strideH, strideW, inputArr, filterArr, outArr, true, true);
	STOP_TIMER("Total Time for Conv");
    uint64_t comm_end = he_conv.io->counter;
    cout << "Total Comm: " << (comm_end - comm_start)/(1.0*(1ULL << 20)) << endl;
}

int main(int argc, char** argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("h", image_h, "Image Height/Width");
    amap.arg("f", filter_h, "Filter Height/Width");
    amap.arg("i", inp_chans, "Input Channels");
    amap.arg("o", out_chans, "Ouput Channels");
    amap.arg("s", stride, "stride");
    amap.arg("pl", pad_l, "Left Padding");
    amap.arg("pr", pad_r, "Right Padding");
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

    cout << "==================================================================" << endl;
    cout << "Role: " << party << " - Bitlength: " << bitlength
        << " - Image: " << image_h << "x" << image_h << "x" << inp_chans
        << " - Filter: " << filter_h << "x" << filter_h << "x" << out_chans
        << "\n- Stride: " << stride << "x" << stride
        << " - Padding: " << pad_l << "x" << pad_r
        << " - # Threads: " << numThreads << endl;
    cout << "==================================================================" << endl;

    io = new NetIO(party==1 ? nullptr:address.c_str(), port);

    ConvField he_conv(party, io);

    Conv(he_conv, image_h, inp_chans, filter_h, out_chans, pad_l, pad_r, stride);

    io->flush();
    return 0;
}
