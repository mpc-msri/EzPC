#pragma once

/*
Authors: Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
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

#pragma once

#include <cryptoTools/Common/Defines.h>
#include <llama/comms.h>

inline void freeDCFKeyPack(DCFKeyPack &key){
    if (!LlamaConfig::dealer->ramdisk) {
        delete[] key.k;
    }
    delete[] key.g;
    // if (!(LlamaConfig::dealer->ramdisk && (key.Bin > 32))) {
        delete[] key.v;
    // }
}

inline void freeDCFKeyPack(DCFET1KeyPack &key){
    if (!LlamaConfig::dealer->ramdisk) {
        delete[] key.k;
    }
}

inline void freeDCFKeyPackPair(std::pair<DCFKeyPack, DCFKeyPack> &keys){
    delete[] keys.first.k;
    delete[] keys.second.k;
    delete[] keys.first.g;
    delete[] keys.first.v;
}

inline void freeDualDCFKeyPack(DualDCFKeyPack &key){
    freeDCFKeyPack(key.dcfKey);
    delete[] key.sb;
}

inline void freeDualDCFKeyPackPair(std::pair<DualDCFKeyPack, DualDCFKeyPack> &keys){
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
    delete[] keys.first.sb;
    delete[] keys.second.sb;
}

inline void freeMatMulKey(MatMulKey &key){
    delete[] key.a;
    delete[] key.b;
    delete[] key.c;
}

inline void freeMatMulKeyPair(std::pair<MatMulKey, MatMulKey> &keys){
    delete[] keys.first.a;
    delete[] keys.first.b;
    delete[] keys.first.c;
    delete[] keys.second.a;
    delete[] keys.second.b;
    delete[] keys.second.c;
}

inline void freeConv2dKey(Conv2DKey &key){
    delete[] key.a;
    delete[] key.b;
    delete[] key.c;
}

inline void freeReluKeyPack(ReluKeyPack &key)
{
    if (!LlamaConfig::dealer->ramdisk) {
        delete[] key.k;
    }
    delete[] key.g;
    if (!(LlamaConfig::dealer->ramdisk && (key.Bin > 32))) {
        delete[] key.v;
    }
}

inline void freeReluKeyPackPair(std::pair<ReluKeyPack,ReluKeyPack> &keys)
{
    delete[] keys.first.k;
    delete[] keys.second.k;
    delete[] keys.first.g;
    delete[] keys.first.v;
    // other key shares g and v, dont delete again
}

inline void freeMaxpoolKeyPack(MaxpoolKeyPack &key)
{
    freeReluKeyPack(key.reluKey);
}

inline void freeMaxpoolKeyPackPair(std::pair<MaxpoolKeyPack,MaxpoolKeyPack> &keys)
{
    delete[] keys.first.reluKey.k;
    delete[] keys.second.reluKey.k;
    delete[] keys.first.reluKey.g;
    delete[] keys.first.reluKey.v;
}

inline void freeARSKeyPack(ARSKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
    if (key.Bout > key.Bin - key.shift) {
        freeDualDCFKeyPack(key.dualDcfKey);
    }
}
inline void freeARSKeyPackPair(std::pair<ARSKeyPack, ARSKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
    if (keys.first.Bout > keys.first.Bin - keys.first.shift) {
        delete[] keys.first.dualDcfKey.sb;
        delete[] keys.second.dualDcfKey.sb;
        delete[] keys.first.dualDcfKey.dcfKey.k;
        delete[] keys.second.dualDcfKey.dcfKey.k;
        delete[] keys.first.dualDcfKey.dcfKey.g;
        delete[] keys.first.dualDcfKey.dcfKey.v;
    }
}

inline void freeReluTruncateKeyPack( ReluTruncateKeyPack &key)
{
    freeDCFKeyPack(key.dcfKeyN);
    freeDCFKeyPack(key.dcfKeyS);
}

inline void freeReluTruncateKeyPackPair(const std::pair<ReluTruncateKeyPack, ReluTruncateKeyPack> &keys)
{
    delete[] keys.first.dcfKeyN.k;
    delete[] keys.second.dcfKeyN.k;
    delete[] keys.first.dcfKeyN.g;
    delete[] keys.first.dcfKeyN.v;

    delete[] keys.first.dcfKeyS.k;
    delete[] keys.second.dcfKeyS.k;
    delete[] keys.first.dcfKeyS.g;
    delete[] keys.first.dcfKeyS.v;
}

inline void freeRelu2RoundKeyPack(Relu2RoundKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

inline void freeRelu2RoundKeyPackPair(const std::pair<Relu2RoundKeyPack, Relu2RoundKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
}

inline void freeSplineKey(SplineKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
    key.p.clear();
    key.e_b.clear();
    key.beta_b.clear();
}

inline void freeSplineKeyPair(std::pair<SplineKeyPack, SplineKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
    keys.first.p.clear();
    keys.second.p.clear();
    keys.first.e_b.clear();
    keys.second.e_b.clear();
    keys.first.beta_b.clear();
    keys.second.beta_b.clear();
}

inline void freeMICKeyPack(MICKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
    delete[] key.z;
}

inline void freeMSNZBKeyPack(MSNZBKeyPack &key)
{
    freeMICKeyPack(key.micKey);
}

inline void freeBulkyLRSKeyPack(BulkyLRSKeyPack &key, int m)
{
    freeDCFKeyPack(key.dcfKeyN);
    delete[] key.z;
    for(int i = 0; i < m; i++) {
        freeDCFKeyPack(key.dcfKeyS[i]);
    }
    delete[] key.dcfKeyS;
}

inline void freeTaylorKeyPack(TaylorKeyPack &key, int m)
{
    freeMSNZBKeyPack(key.msnzbKey);
    freeBulkyLRSKeyPack(key.lrsKeys[0], m);
    freeBulkyLRSKeyPack(key.lrsKeys[1], m);
}

inline void freeMaxpoolDoubleKeyPack(MaxpoolDoubleKeyPack &key)
{
    freeRelu2RoundKeyPack(key.reluKey);
}

inline void freeMaxpoolDoubleKeyPackPair(std::pair<MaxpoolDoubleKeyPack,MaxpoolDoubleKeyPack> &keys)
{
    delete[] keys.first.reluKey.dcfKey.k;
    delete[] keys.second.reluKey.dcfKey.k;
}

inline void freeFixToFloatKeyPack(FixToFloatKeyPack &key)
{
    freeMICKeyPack(key.micKey);
}

inline void freeFixToFloatKeyPackPair(std::pair<FixToFloatKeyPack, FixToFloatKeyPack> &keys)
{
    delete[] keys.first.micKey.dcfKey.k;
    delete[] keys.second.micKey.dcfKey.k;
    delete[] keys.first.micKey.dcfKey.g;
    delete[] keys.first.micKey.dcfKey.v;
}

inline void freeFloatToFixKeyPack(FloatToFixKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

inline void freeFloatToFixKeyPackPair(std::pair<FloatToFixKeyPack, FloatToFixKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

inline void freeReluExtendKeyPack(ReluExtendKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

inline void freeReluExtendKeyPackPair(std::pair<ReluExtendKeyPack, ReluExtendKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

inline void freeSignExtend2KeyPack(SignExtend2KeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

inline void freeSignExtend2KeyPackPair(std::pair<SignExtend2KeyPack, SignExtend2KeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}
