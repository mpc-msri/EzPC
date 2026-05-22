// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
Authors: Deepak Kumaraswamy, Kanav Gupta
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

#include <string>
#include <iostream>
#include <llama/keypack.h>
#include <llama/array.h>
#include <llama/assert.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>

#define DEALER 1
#define SERVER 2
#define CLIENT 3

typedef enum BufType
{
    BUF_FILE,
    BUF_SOCKET,
    BUF_MEM
} BufType;

class KeyBuf
{
public:
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;
    BufType t;
    virtual void sync() {}
    virtual void read(char *buf, int bytes) = 0;
    virtual char *read(int bytes) = 0;
    virtual void write(char *buf, int bytes) = 0;
    virtual void close() = 0;
    bool isMem() { return t == BUF_MEM; }
};

typedef enum FileMode
{
    F_RD_ONLY,
    F_WR_ONLY
} FileMode;

class FileBuf : public KeyBuf
{
public:
    std::fstream file;

    FileBuf(std::string filename, FileMode f)
    {
        printf("Opening file=%s, mode=%d\n", filename.data(), f);
        this->t = BUF_FILE;
        if (f == F_WR_ONLY)
            this->file.open(filename, std::ios::out | std::ios::binary);
        else
            this->file.open(filename, std::ios::in | std::ios::binary);
    }

    void read(char *buf, int bytes)
    {
        this->file.read(buf, bytes);
        bytesReceived += bytes;
    }

    char *read(int bytes)
    {
        char *newBuf = new char[bytes];
        this->read(newBuf, bytes);
        return newBuf;
    }

    void write(char *buf, int bytes)
    {
        this->file.write(buf, bytes);
        bytesSent += bytes;
    }

    void close()
    {
        this->file.close();
    }
};

class SocketBuf : public KeyBuf
{
public:
    int sendsocket, recvsocket;

    SocketBuf(std::string ip, int port, bool onlyRecv);
    SocketBuf(int sendsocket, int recvsocket) : sendsocket(sendsocket), recvsocket(recvsocket)
    {
        this->t = BUF_SOCKET;
    }
    void sync();
    void read(char *buf, int bytes);
    char *read(int bytes);
    void write(char *buf, int bytes);
    void close();
};

class MemBuf : public KeyBuf
{
public:
    char **memBufPtr;
    char *startPtr;

    MemBuf(char **mBufPtr)
    {
        this->t = BUF_MEM;
        memBufPtr = mBufPtr;
        startPtr = *mBufPtr;
    }

    void read(char *buf, int bytes)
    {
        memcpy(buf, *memBufPtr, bytes);
        *memBufPtr += bytes;
        bytesReceived += bytes;
    }

    char *read(int bytes)
    {
        char *newBuf = *memBufPtr;
        *memBufPtr += bytes;
        bytesReceived += bytes;
        return newBuf;
    }

    void write(char *buf, int bytes)
    {
        memcpy(*memBufPtr, buf, bytes);
        *memBufPtr += bytes;
        bytesSent += bytes;
    }

    void close()
    {
        // do nothing yet
    }
};

class Peer
{
public:
    KeyBuf *keyBuf;

    Peer(std::string ip, int port)
    {
        keyBuf = new SocketBuf(ip, port, false);
    }

    Peer(int sendsocket, int recvsocket)
    {
        keyBuf = new SocketBuf(sendsocket, recvsocket);
    }

    Peer(std::string filename)
    {
        keyBuf = new FileBuf(filename, F_WR_ONLY);
    }

    Peer(char **mBufPtr)
    {
        keyBuf = new MemBuf(mBufPtr);
    }

    inline uint64_t bytesSent()
    {
        return keyBuf->bytesSent;
    }

    inline uint64_t bytesReceived()
    {
        return keyBuf->bytesReceived;
    }

    void inline zeroBytesSent()
    {
        keyBuf->bytesSent = 0;
    }

    void inline zeroBytesReceived()
    {
        keyBuf->bytesReceived = 0;
    }

    void close();

    void send_ge(const GroupElement &g, int bw);
    void send_ge_array(const GroupElement *g, int size);

    void send_block(const osuCrypto::block &b);

    void send_mask(const GroupElement &g);

    void send_input(const GroupElement &g);

    void send_batched_input(GroupElement *g, int size, int bw);

    void send_mult_key(const MultKey &k);

    void send_square_key(const SquareKey &k);

    void send_matmul_key(const MatMulKey &k);

    void send_new_mult_key(const MultKeyNew &k, int bw1, int bw2);

    void send_conv2d_key(const Conv2DKey &k);

    void send_conv3d_key(const Conv3DKey &k);

    void send_dcf_keypack(const DCFKeyPack &kp);

    void send_dpf_keypack(const DPFKeyPack &kp);

    void send_dpfet_keypack(const DPFETKeyPack &kp);

    void send_ddcf_keypack(const DualDCFKeyPack &kp);

    void send_relu_key(const ReluKeyPack &kp);

    void send_maxpool_key(const MaxpoolKeyPack &kp);

    void send_maxpool_double_key(const MaxpoolDoubleKeyPack &kp);

    void send_scmp_keypack(const ScmpKeyPack &kp);

    void send_pubdiv_key(const PublicDivKeyPack &kp);

    void send_ars_key(const ARSKeyPack &kp);

    void send_spline_key(const SplineKeyPack &kp);

    void send_signedpubdiv_key(const SignedPublicDivKeyPack &kp);

    void send_publicIC_key(const PublicICKeyPack &kp);

    void send_relu_truncate_key(const ReluTruncateKeyPack &kp);

    void send_relu_2round_key(const Relu2RoundKeyPack &kp);

    void send_select_key(const SelectKeyPack &kp);

    void send_bulkylrs_key(const BulkyLRSKeyPack &kp, int bl, int m);

    void send_taylor_key(const TaylorKeyPack &kp, int bl, int m);

    void send_bitwise_and_key(const BitwiseAndKeyPack &kp);

    void send_mic_key(const MICKeyPack &kp, int bin, int bout, int m);

    void send_fix_to_float_key(const FixToFloatKeyPack &kp, int bl);

    void send_float_to_fix_key(const FloatToFixKeyPack &kp, int bl);

    void send_relu_extend_key(const ReluExtendKeyPack &kp, int bin, int bout);

    void send_sign_extend2_key(const SignExtend2KeyPack &kp, int bin, int bout);

    void send_triple_key(const TripleKeyPack &kp);

    void send_edabits_prtrunc_key(const EdabitsPrTruncKeyPack &kp, int bw);

    void send_pubcmp_key(const PubCmpKeyPack &kp);

    void send_clip_key(const ClipKeyPack &kp);

    void send_lut_key(const LUTKeyPack &kp);

    void send_lutdpfet_key(const LUTDPFETKeyPack &kp);

    void send_f2bf16_key(const F2BF16KeyPack &kp);

    void send_truncate_reduce_key(const TruncateReduceKeyPack &kp);

    void send_lutss_key(const LUTSSKeyPack &kp);

    void send_sloth_drelu_key(const SlothDreluKeyPack &kp);

    void send_wrap_dpf_key(const WrapDPFKeyPack &kp);

    void send_wrap_ss_key(const WrapSSKeyPack &kp);

    void send_sloth_lrs_key(const SlothLRSKeyPack &kp);

    void send_sloth_sign_extend_key(const SlothSignExtendKeyPack &kp);

    void send_uint8_array(const uint8_t *data, int size);

    void recv_uint8_array(uint8_t *data, int size);

    void sync();

    GroupElement recv_input();

    void recv_batched_input(uint64_t *g, int size, int bw);
};

Peer *waitForPeer(int port);

class Dealer
{
public:
    KeyBuf *keyBuf;

    Dealer(std::string ip, int port);

    Dealer(std::string filename)
    {
        keyBuf = new FileBuf(filename, F_RD_ONLY);
    }

    Dealer(char **mBufPtr)
    {
        keyBuf = new MemBuf(mBufPtr);
    }

    inline uint64_t bytesReceived()
    {
        return keyBuf->bytesReceived;
    }

    void close();

    GroupElement recv_mask();

    MultKey recv_mult_key();

    SquareKey recv_square_key();

    osuCrypto::block recv_block();

    osuCrypto::block *recv_block_array(int numBlocks);

    GroupElement recv_ge(int bw);

    GroupElement *recv_ge_array(int bw, int size);

    void recv_ge_array(const GroupElement *g, int size);

    void recv_ge_array(int bw, int size, GroupElement *arr);

    DCFKeyPack recv_dcf_keypack(int Bin, int Bout, int groupSize);

    DPFKeyPack recv_dpf_keypack(int bin, int bout);

    DPFETKeyPack recv_dpfet_keypack(int bin);

    DualDCFKeyPack recv_ddcf_keypack(int Bin, int Bout, int groupSize);

    MatMulKey recv_matmul_key(int bin, int bout, int s1, int s2, int s3);

    Conv2DKey recv_conv2d_key(int bin, int bout, int64_t N, int64_t H, int64_t W,
                              int64_t CI, int64_t FH, int64_t FW,
                              int64_t CO, int64_t zPadHLeft,
                              int64_t zPadHRight, int64_t zPadWLeft,
                              int64_t zPadWRight, int64_t strideH,
                              int64_t strideW);

    Conv3DKey recv_conv3d_key(int bin, int bout, int64_t N, int64_t D, int64_t H, int64_t W,
                              int64_t CI, int64_t FD, int64_t FH, int64_t FW, int64_t CO,
                              int64_t zPadDLeft, int64_t zPadDRight,
                              int64_t zPadHLeft, int64_t zPadHRight,
                              int64_t zPadWLeft, int64_t zPadWRight,
                              int64_t strideD, int64_t strideH, int64_t strideW);

    ReluKeyPack recv_relu_key(int Bin, int Bout);

    MaxpoolKeyPack recv_maxpool_key(int Bin, int Bout);

    MaxpoolDoubleKeyPack recv_maxpool_double_key(int Bin, int Bout);

    ScmpKeyPack recv_scmp_keypack(int Bin, int Bout);

    PublicDivKeyPack recv_pubdiv_key(int Bin, int Bout);

    ARSKeyPack recv_ars_key(int Bin, int Bout, int shift);

    MultKeyNew recv_new_mult_key(int Bin, int Bout);

    SplineKeyPack recv_spline_key(int Bin, int Bout, int numPoly, int degree);

    SignedPublicDivKeyPack recv_signedpubdiv_key(int Bin, int Bout);

    PublicICKeyPack recv_publicIC_key(int Bin, int Bout);

    ReluTruncateKeyPack recv_relu_truncate_key(int Bin, int Bout, int s);

    Relu2RoundKeyPack recv_relu_2round_key(int effectiveBin, int Bin);

    SelectKeyPack recv_select_key(int Bin);

    TaylorKeyPack recv_taylor_key(int bl, int m, int sf);

    BulkyLRSKeyPack recv_bulkylrs_key(int bl, int m, uint64_t *scales);

    BitwiseAndKeyPack recv_bitwise_and_key();

    MICKeyPack recv_mic_key(int bin, int bout, int m);

    FixToFloatKeyPack recv_fix_to_float_key(int bl);

    FloatToFixKeyPack recv_float_to_fix_key(int bl);

    ReluExtendKeyPack recv_relu_extend_key(int Bin, int Bout);

    SignExtend2KeyPack recv_sign_extend2_key(int Bin, int Bout);

    TripleKeyPack recv_triple_key(int bw, int64_t na, int64_t nb, int64_t nc);

    EdabitsPrTruncKeyPack recv_edabits_prtrunc_key(int bw);

    PubCmpKeyPack recv_pubcmp_key(int bin);

    ClipKeyPack recv_clip_key(int bin);

    LUTKeyPack recv_lut_key(int bin, int bout);

    LUTDPFETKeyPack recv_lutdpfet_key(int bin, int bout);

    F2BF16KeyPack recv_f2bf16_key(int bin);

    TruncateReduceKeyPack recv_truncate_reduce_key(int bin, int shift);

    LUTSSKeyPack recv_lutss_key(int bin, int bout);

    SlothDreluKeyPack recv_slothdrelu_key(int bin);

    WrapDPFKeyPack recv_wrap_dpf_key(int bin);

    WrapSSKeyPack recv_wrap_ss_key(int bin);

    SlothLRSKeyPack recv_sloth_lrs_key(int bin, int shift);

    SlothSignExtendKeyPack recv_sloth_sign_extend_key(int bin, int bout);
};
