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

class Peer {
public:
    int sendsocket, recvsocket;
    bool useFile = false;
    std::fstream file;
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;
    

    Peer(std::string ip, int port);
    Peer(int sendsocket, int recvsocket) {
        this->sendsocket = sendsocket;
        this->recvsocket = recvsocket;
    }
    Peer(std::string filename) {
        this->useFile = true;
        this->file.open(filename, std::ios::out | std::ios::binary);
    }

    void close();

    void send_ge(const GroupElement &g, int bw);
    void send_ge_array(const GroupElement *g, int size);

    void send_block(const osuCrypto::block &b);

    void send_mask(const GroupElement &g);

    void send_input(const GroupElement &g);

    void send_batched_input(GroupElement *g, int size, int bw);

    void send_mult_key(const MultKey &k);

    void send_matmul_key(const MatMulKey &k);

    void send_new_mult_key(const MultKeyNew &k, int bw1, int bw2);

    void send_conv2d_key(const Conv2DKey &k);

    void send_conv3d_key(const Conv3DKey &k);

    void send_dcf_keypack(const DCFKeyPack &kp);

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

    void send_uint8_array(const uint8_t *data, int size);

    void recv_uint8_array(uint8_t *data, int size);

    void sync();

    GroupElement recv_input();

    void recv_batched_input(uint64_t *g, int size, int bw);

};

Peer* waitForPeer(int port);

class Dealer {
public:
    int consocket;
    bool useFile = true;
    std::fstream file;
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;
    bool ramdisk =true;
    char *ramdiskBuffer;
    char *ramdiskStart;
    int ramdiskSize;
    bool ramdisk_path = false;

    Dealer(std::string ip, int port);

    Dealer(std::string filename, bool ramdisk,bool ramdisk_path) {
        this->useFile = true;
        this->ramdisk = ramdisk;
        this->ramdisk_path = ramdisk_path;
        if (ramdisk && ramdisk_path) {
            int fd = open(filename.c_str(), O_RDWR | O_CREAT, 0);
            struct stat sb;
            fstat(fd, &sb);
            std::cerr << "Key Size: " << sb.st_size << " bytes" << "\n";
            int advise=posix_fadvise(fd, 0, sb.st_size, POSIX_FADV_WILLNEED);
            ramdiskSize = sb.st_size;
            ramdiskBuffer = (char*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            ramdiskStart = ramdiskBuffer;
            std::cout << "RAMDISK: " << (int *)ramdiskBuffer << "\n";
            ::close(fd);
        }
        else {
            this->file.open(filename, std::ios::in | std::ios::binary);
        }
    }

    void close();

    GroupElement recv_mask();

    MultKey recv_mult_key();

    osuCrypto::block recv_block();

    GroupElement recv_ge(int bw);

    void recv_ge_array(const GroupElement *g, int size);

    void recv_ge_array(int bw, int size, GroupElement *arr);

    DCFKeyPack recv_dcf_keypack(int Bin, int Bout, int groupSize);

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

};
