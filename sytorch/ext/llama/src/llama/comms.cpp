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

#include <llama/comms.h>
#include <llama/assert.h>

using namespace LlamaConfig;

Peer::Peer(std::string ip, int port) {
    std::cerr << "trying to connect with server...";
    {   
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        while (1) {
        recvsocket = socket(AF_INET, SOCK_STREAM, 0);

        if (connect(recvsocket, (struct sockaddr *)&addr,
                    sizeof(struct sockaddr)) == 0) {
          break;
        }

        ::close(recvsocket);
        usleep(1000);
      }
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    sleep(1);
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port+3);
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        while(1){
            sendsocket = socket(AF_INET, SOCK_STREAM, 0);
            if (sendsocket < 0) {
                perror("socket");
                exit(1);
            }
            if (connect(sendsocket, (struct sockaddr *) &addr, sizeof(addr)) == 0) {
                break;
            }
            ::close(sendsocket);
            usleep(1000);
        }
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    std::cerr << "connected" << "\n";

}

void Peer::close() {
    if (useFile) {
        file.close();
    }
    else {
        ::close(sendsocket);
        ::close(recvsocket);
    }
}

Peer* waitForPeer(int port) {
    int sendsocket, recvsocket;
    std::cerr << "waiting for connection from client...";
    
    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY);       /* set our address to any interface */
        serv.sin_port = htons(port); /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                    sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0) {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0) {
            perror("error: listen");
            exit(1);
        }
        sendsocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }

    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY);       /* set our address to any interface */
        serv.sin_port = htons(port+3); /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                    sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0) {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0) {
            perror("error: listen");
            exit(1);
        }
        recvsocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }
    
    std::cerr << "connected" << "\n";
    return new Peer(sendsocket, recvsocket);
}


void Peer::send_ge(const GroupElement &g, int bw) {
    if (bw > 32) {
        char *buf = (char *)(&g);
        if (useFile) {
            this->file.write(buf, 8);
        } else {
            send(sendsocket, buf, 8, 0);
        }
        bytesSent += 8;
    }
    else if (bw > 16) {
        char *buf = (char *)(&g);
        if (useFile) {
            this->file.write(buf, 4);
        } else {
            send(sendsocket, buf, 4, 0);
        }
        bytesSent += 4;
    }
    else if (bw > 8) {
        char *buf = (char *)(&g);
        if (useFile) {
            this->file.write(buf, 2);
        } else {
            send(sendsocket, buf, 2, 0);
        }
        bytesSent += 2;
    }
    else {
        char *buf = (char *)(&g);
        if (useFile) {
            this->file.write(buf, 1);
        } else {
            send(sendsocket, buf, 1, 0);
        }
        bytesSent += 1;
    }
}


void Peer::send_ge_array(const GroupElement *g, int size) {
    char *buf = (char *)(g);
    if (useFile) {
        this->file.write(buf, 8*size);
    } else {
        send(sendsocket, buf, 8*size, 0);
    }
    bytesSent += (8*size);
}

void Peer::send_block(const osuCrypto::block &b) {
    char *buf = (char *)(&b);
    if (useFile) {
        this->file.write(buf, sizeof(osuCrypto::block));
    } else {
        send(sendsocket, buf, sizeof(osuCrypto::block), 0);
    }
    bytesSent += sizeof(osuCrypto::block);
}

void Peer::send_mask(const GroupElement &g) {
    send_ge(g, 64);
}

void Peer::send_input(const GroupElement &g) {
    send_ge(g, 64);
}

void Peer::send_batched_input(GroupElement *g, int size, int bw)
{
    if (bw > 32) {
        uint64_t *temp = new uint64_t[size];
        for (int i = 0; i < size; i++) {
            temp[i] = g[i];
        }
        char *buf = (char *)(temp);
        if (useFile) {
            this->file.write(buf, 8*size);
        } else {
            send(sendsocket, buf, 8*size, 0);
        }
        delete[] temp;
        bytesSent += 8*size;
    }
    else if (bw > 16) {
        uint32_t *temp = new uint32_t[size];
        for (int i = 0; i < size; i++) {
            temp[i] = (uint32_t)g[i];
        }
        char *buf = (char *)(temp);
        if (useFile) {
            this->file.write(buf, 4*size);
        } else {
            send(sendsocket, buf, 4*size, 0);
        }
        delete[] temp;
        bytesSent += 4*size;
    }
    else if (bw > 8) {
        uint16_t *temp = new uint16_t[size];
        for (int i = 0; i < size; i++) {
            temp[i] = (uint16_t)g[i];
        }
        char *buf = (char *)(temp);
        if (useFile) {
            this->file.write(buf, 2*size);
        } else {
            send(sendsocket, buf, 2*size, 0);
        }
        delete[] temp;
        bytesSent += 2*size;
    }
    else {
        uint8_t *temp = new uint8_t[size];
        for (int i = 0; i < size; i++) {
            temp[i] = (uint8_t)g[i];
        }
        char *buf = (char *)(temp);
        if (useFile) {
            this->file.write(buf, size);
        } else {
            send(sendsocket, buf, size, 0);
        }
        delete[] temp;
        bytesSent += size;
    }
}

void Peer::recv_batched_input(uint64_t *g, int size, int bw)
{
    if (bw > 32) {
        if (useFile) {
            this->file.read((char *)g, 8*size);
        } else {
            recv(recvsocket, (char *)g, 8*size, MSG_WAITALL);
        }
        bytesReceived += 8*size;
    }
    else if (bw > 16) {
        uint32_t *tmp = new uint32_t[size];
        if (useFile) {
            this->file.read((char *)tmp, 4*size);
        } else {
            recv(recvsocket, (char *)tmp, 4*size, MSG_WAITALL);
        }
        for (int i = 0; i < size; i++) {
            g[i] = tmp[i];
        }
        delete[] tmp;
        bytesReceived += 4*size;
    }
    else if (bw > 8) {
        uint16_t *tmp = new uint16_t[size];
        if (useFile) {
            this->file.read((char *)tmp, 2*size);
        } else {
            recv(recvsocket, (char *)tmp, 2*size, MSG_WAITALL);
        }
        for (int i = 0; i < size; i++) {
            g[i] = tmp[i];
        }
        delete[] tmp;
        bytesReceived += 2*size;
    }
    else {
        uint8_t *tmp = new uint8_t[size];
        if (useFile) {
            this->file.read((char *)tmp, size);
        } else {
            recv(recvsocket, (char *)tmp, size, MSG_WAITALL);
        }
        for (int i = 0; i < size; i++) {
            g[i] = tmp[i];
        }
        delete[] tmp;
        bytesReceived += size;
    }
}

void Peer::send_mult_key(const MultKey &k) {
    char *buf = (char *)(&k);
    if (useFile) {
        this->file.write(buf, sizeof(MultKey));
    } else {
        send(sendsocket, buf, sizeof(MultKey), 0);
    }
    bytesSent += sizeof(MultKey);
}

void Peer::send_matmul_key(const MatMulKey &k) {
    int s1 = k.s1;
    int s2 = k.s2;
    int s3 = k.s3;
    
    for(int i = 0; i < s1; i++) {
        for(int j = 0; j < s2; j++) {
            send_ge(Arr2DIdx(k.a, s1, s2, i, j), k.Bin);
        }
    }

    for(int i = 0; i < s2; i++) {
        for(int j = 0; j < s3; j++) {
            send_ge(Arr2DIdx(k.b, s2, s3, i, j), k.Bin);
        }
    }

    for(int i = 0; i < s1; i++) {
        for(int j = 0; j < s3; j++) {
            send_ge(Arr2DIdx(k.c, s1, s3, i, j), k.Bout);
        }
    }
}

void Peer::send_conv2d_key(const Conv2DKey &k) {
    int N = k.N;
    int H = k.H;
    int W = k.W;
    int CO = k.CO;
    int CI = k.CI;
    int FH = k.FH;
    int FW = k.FW;
    int zPadHLeft = k.zPadHLeft;
    int zPadHRight = k.zPadHRight; 
    int zPadWLeft = k.zPadWLeft;
    int zPadWRight = k.zPadWRight;;
    int strideH = k.strideH;
    int strideW = k.strideW;

    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;
    

    for(int n = 0; n < N; ++n) {
        for(int h = 0; h < H; ++h) {
            for(int w = 0; w < W; ++w) {
                for(int ci = 0; ci < CI; ++ci) {
                    send_ge(Arr4DIdx(k.a, N, H, W, CI, n, h, w, ci), k.Bin);
                }
            }
        }
    }

    for(int fh = 0; fh < FH; ++fh) {
        for(int fw = 0; fw < FW; ++fw) {
            for(int ci = 0; ci < CI; ++ci) {
                for(int co = 0; co < CO; ++co) {
                    send_ge(Arr4DIdx(k.b, FH, FW, CI, CO, fh, fw, ci, co), k.Bin);
                }
            }
        }
    }

    GroupElement *c = k.c;
    int Bout = k.Bout;
    for(int i = 0; i < d0; ++i) {
        for(int j = 0; j < d1; ++j) {
            for(int k = 0; k < d2; ++k) {
                for(int l = 0; l < d3; ++l) {
                    send_ge(Arr4DIdx(c, d0, d1, d2, d3, i, j, k, l), Bout);
                }
            }
        }
    }
}

void Peer::send_conv3d_key(const Conv3DKey &k) {
    int N = k.N;
    int D = k.D;
    int H = k.H;
    int W = k.W;
    int CO = k.CO;
    int CI = k.CI;
    int FD = k.FD;
    int FH = k.FH;
    int FW = k.FW;
    int zPadDLeft = k.zPadDLeft;
    int zPadDRight = k.zPadDRight;
    int zPadHLeft = k.zPadHLeft;
    int zPadHRight = k.zPadHRight; 
    int zPadWLeft = k.zPadWLeft;
    int zPadWRight = k.zPadWRight;;
    int strideH = k.strideH;
    int strideW = k.strideW;
    int strideD = k.strideD;

    int d0 = N;
    int d1 = ((D - FD + (zPadDLeft + zPadDRight)) / strideD) + 1;
    int d2 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d3 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d4 = CO;

    for(size_t i = 0; i < N * D * H * W * CI; ++i) {
        send_ge(k.a[i], k.Bin);
    }

    for(size_t i = 0; i < FD * FH * FW * CI * CO; ++i) {
        send_ge(k.b[i], k.Bin);
    }

    for(size_t i = 0; i < d0 * d1 * d2 * d3 * d4; ++i) {
        send_ge(k.c[i], k.Bout);
    }
}

void Peer::send_dcf_keypack(const DCFKeyPack &kp) {
    for (int i = 0; i < kp.Bin + 1; ++i) {
        send_block(kp.k[i]);
    }
    for (int i = 0; i < kp.groupSize; ++i) {
        send_ge(kp.g[i], kp.Bout);
    }
    for (int i = 0; i < kp.groupSize * kp.Bin; ++i) {
        send_ge(kp.v[i], kp.Bout);
    }
    send_ge(42, 64);
}

void Peer::send_ddcf_keypack(const DualDCFKeyPack &kp) {
    send_dcf_keypack(kp.dcfKey);
    for (int i = 0; i < kp.groupSize; ++i) {
        send_ge(kp.sb[i], kp.Bout);
    }
}

void Peer::send_new_mult_key(const MultKeyNew &k, int bw1, int bw2)
{
    send_ge(k.a, bw1 + bw2);
    send_ge(k.b, bw1 + bw2);
    send_ge(k.c, bw1 + bw2);
    send_dcf_keypack(k.k1);
    send_dcf_keypack(k.k2);
    send_dcf_keypack(k.k3);
    send_dcf_keypack(k.k4);
}

void Peer::send_relu_key(const ReluKeyPack &kp) {
    int Bin = kp.Bin;
    int groupSize = 2;
    for(int i = 0; i < Bin + 1; ++i) {
        send_block(kp.k[i]);
    }
    for(int i = 0; i < groupSize; ++i) {
        send_ge(kp.g[i], kp.Bout);
    }
    for(int i = 0; i < Bin * groupSize; ++i) {
        send_ge(kp.v[i], kp.Bout);
    }
    send_ge(kp.e_b0, kp.Bout);
    send_ge(kp.e_b1, kp.Bout);
    send_ge(kp.beta_b0, kp.Bout);
    send_ge(kp.beta_b1, kp.Bout);
    send_ge(kp.r_b, kp.Bout);
    send_ge(kp.drelu, 1);
}

void Peer::send_maxpool_key(const MaxpoolKeyPack &kp) {
    send_relu_key(kp.reluKey);
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_maxpool_double_key(const MaxpoolDoubleKeyPack &kp) {
    send_relu_2round_key(kp.reluKey);
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_scmp_keypack(const ScmpKeyPack &kp) {
    send_ddcf_keypack(kp.dualDcfKey);
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_pubdiv_key(const PublicDivKeyPack &kp) {
    send_ddcf_keypack(kp.dualDcfKey);
    send_scmp_keypack(kp.scmpKey);
    send_ge(kp.zb, kp.Bout);
}

void Peer::send_ars_key(const ARSKeyPack &kp) {
    if (!LlamaConfig::stochasticT)
        send_dcf_keypack(kp.dcfKey);
    if (kp.Bout > kp.Bin - kp.shift) {
        send_ddcf_keypack(kp.dualDcfKey);
    }
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_spline_key(const SplineKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    for(auto &pi: kp.p) {
        send_ge(pi, kp.Bin);
    }

    for(auto &row: kp.e_b) {
        for(auto &e: row) {
            send_ge(e, kp.Bout);
        }
    }

    for(auto &b: kp.beta_b) {
        send_ge(b, kp.Bout);
    }

    send_ge(kp.r_b, kp.Bout);
}

void Peer::send_signedpubdiv_key(const SignedPublicDivKeyPack &kp) {
    send_ge(kp.d, kp.Bin);
    send_dcf_keypack(kp.dcfKey);
    send_publicIC_key(kp.publicICkey);
    send_scmp_keypack(kp.scmpKey);
    send_ge(kp.A_share, kp.Bin);
    send_ge(kp.corr_share, kp.Bout);
    send_ge(kp.B_share, kp.Bout);
    send_ge(kp.rdiv_share, kp.Bout);
    send_ge(kp.rout_temp_share, kp.Bout);
    send_ge(kp.rout_share, kp.Bout);
}

void Peer::send_publicIC_key(const PublicICKeyPack &kp) {
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.zb, kp.Bout);
}

void Peer::send_relu_truncate_key(const ReluTruncateKeyPack &kp) {
    send_dcf_keypack(kp.dcfKeyN);
    if (!LlamaConfig::stochasticRT)
        send_dcf_keypack(kp.dcfKeyS);
    send_ge(kp.zTruncate, kp.Bin);
    send_ge(kp.a, kp.Bin);
    send_ge(kp.b, kp.Bin);
    send_ge(kp.c, kp.Bin);
    send_ge(kp.d1, kp.Bin);
    send_ge(kp.d2, kp.Bin);
}

void Peer::send_relu_2round_key(const Relu2RoundKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.a, kp.Bin);
    send_ge(kp.b, kp.Bin);
    send_ge(kp.c, kp.Bin);
    send_ge(kp.d1, kp.Bin);
    send_ge(kp.d2, kp.Bin);
}

void Peer::send_select_key(const SelectKeyPack &kp) {
    send_ge(kp.a, kp.Bin);
    send_ge(kp.b, kp.Bin);
    send_ge(kp.c, kp.Bin);
    send_ge(kp.d1, kp.Bin);
    send_ge(kp.d2, kp.Bin);
}

void Peer::send_bulkylrs_key(const BulkyLRSKeyPack &kp, int bl, int m) {
    send_dcf_keypack(kp.dcfKeyN);
    for(int i = 0; i < m; ++i)
    {
        send_dcf_keypack(kp.dcfKeyS[i]);
        send_ge(kp.z[i], bl);
    }
    send_ge(kp.out, bl);
}

void Peer::send_taylor_key(const TaylorKeyPack &kp, int bl, int m) {
    send_dcf_keypack(kp.msnzbKey.micKey.dcfKey);
    for(int i = 0; i < m; ++i)
    {
        send_ge(kp.msnzbKey.micKey.z[i], bl);
    }
    send_ge(kp.msnzbKey.r, bl);
    send_ge(kp.squareKey.a, bl);
    send_ge(kp.squareKey.b, bl);
    send_ge(kp.privateScaleKey.rin, bl);
    send_ge(kp.privateScaleKey.rout, bl);
    send_bulkylrs_key(kp.lrsKeys[0], bl, m);
    send_bulkylrs_key(kp.lrsKeys[1], bl, m);
    send_ge(69, bl);
}

void Peer::send_bitwise_and_key(const BitwiseAndKeyPack &kp)
{
    send_ge(kp.t[0], 64);
    send_ge(kp.t[1], 64);
    send_ge(kp.t[2], 64);
    send_ge(kp.t[3], 64);
}

void Peer::send_mic_key(const MICKeyPack &kp, int bin, int bout, int m)
{
    send_dcf_keypack(kp.dcfKey);
    for(int i = 0; i < m; ++i)
    {
        send_ge(kp.z[i], bout);
    }
}

void Peer::send_fix_to_float_key(const FixToFloatKeyPack &kp, int bl)
{
    send_mic_key(kp.micKey, bl, bl, 2*bl);
    send_ge(kp.rs, 1);
    send_ge(kp.rpow, bl);
    send_ge(kp.ry, bl);
    send_select_key(kp.selectKey);
    send_ge(kp.rm, bl);
}

void Peer::send_float_to_fix_key(const FloatToFixKeyPack &kp, int bl)
{
    send_ge(kp.rm, 24);
    send_ge(kp.re, 10);
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rw, 1);
    send_ge(kp.rt, bl);
    send_select_key(kp.selectKey);
    for(int i = 0; i < 1024; ++i) {
        send_ge(kp.p[i], bl);
    }
    for(int i = 0; i < 1024; ++i) {
        send_ge(kp.q[i], bl);
    }
}

void Peer::send_relu_extend_key(const ReluExtendKeyPack &kp, int bin, int bout)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rd, 2);
    send_ge(kp.rw, 2);
    for(int i = 0; i < 4; ++i)
        send_ge(kp.p[i], bout);
    for(int i = 0; i < 2; ++i)
        send_ge(kp.q[i], bout);
}

void Peer::send_sign_extend2_key(const SignExtend2KeyPack &kp, int bin, int bout)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rw, 1);
    send_ge(kp.p[0], bout);
    send_ge(kp.p[1], bout);
}

void Peer::send_triple_key(const TripleKeyPack &kp)
{
    for(size_t i = 0; i < kp.na; ++i) {
        send_ge(kp.a[i], kp.bw);
    }

    for(size_t i = 0; i < kp.nb; ++i) {
        send_ge(kp.b[i], kp.bw);
    }

    for(size_t i = 0; i < kp.nc; ++i) {
        send_ge(kp.c[i], kp.bw);
    }
}

GroupElement Peer::recv_input() {
    char buf[8];
    if (useFile) {
        std::cerr << "Can't recv from peer in file mode\n";
        exit(1);
    } else {
        recv(recvsocket, buf, 8, MSG_WAITALL);
    }
    GroupElement g =*(uint64_t *)buf;
    bytesReceived += 8;
    return g;
}

void Peer::send_uint8_array(const uint8_t *data, int size)
{
    always_assert(size == send(sendsocket, data, size, 0));
    bytesSent += size;
}

void Peer::recv_uint8_array(uint8_t *data, int size)
{
    always_assert(size == recv(recvsocket, data, size, MSG_WAITALL));
    bytesReceived += size;
}

Dealer::Dealer(std::string ip, int port) {
    this->consocket = socket(AF_INET, SOCK_STREAM, 0);
    if (consocket < 0) {
        perror("socket");
        exit(1);
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip.c_str());
    if (connect(consocket, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("connect");
        exit(1);
    }
}

void Dealer::close() {
    if (useFile) {
        if (!ramdisk && ramdisk_path) {
            file.close();
        }
        else {
            // std::cout << (int)(ramdiskBuffer - ramdiskStart) << "bytes read" << "\n";
            // always_assert(ramdiskBuffer - ramdiskStart == ramdiskSize);
        }
    }
    else {
        ::close(consocket);
    }
}

GroupElement Dealer::recv_mask() {
    char buf[8];
    if (useFile) {
        if (ramdisk && ramdisk_path) {
            std::cout<<"ramdiskBuffer: "<<(uint64_t)ramdiskBuffer<<"\n";
            GroupElement g = *(uint64_t *)ramdiskBuffer;
            ramdiskBuffer += 8;
            bytesReceived += 8;
            return g;
        }
        this->file.read(buf, 8);
        //std::cout << "dealer recv mask" << "\n";
    } else {
       // recv(consocket, buf, 8, MSG_WAITALL);
        std::cout << "dealer recv mask" << "\n";
    }
    GroupElement g = *(uint64_t *)buf;
    bytesReceived += 8;
    return g;
}

MultKey Dealer::recv_mult_key() {
    char buf[sizeof(MultKey)];
    if (useFile) {
        if (ramdisk && ramdisk_path) {
             std::cout<<"ramdiskBuffer ,multikey: "<<(uint64_t)ramdiskBuffer<<"\n";
            MultKey k=(*(MultKey *)ramdiskBuffer);
            ramdiskBuffer += sizeof(MultKey);
            bytesReceived += sizeof(MultKey);
            return k;
        }
        this->file.read(buf, sizeof(MultKey));
        //std::cout<< "dealer recv mult key" << "\n";
    } else {
        //recv(consocket, buf, sizeof(MultKey), MSG_WAITALL);
        std::cout << "dealer recv mask" << "\n";
    }
    MultKey k(*(MultKey *)buf);
    bytesReceived += sizeof(MultKey);
    return k;
}

osuCrypto::block Dealer::recv_block() {
    char buf[sizeof(osuCrypto::block)];
    if (useFile) {
        if (ramdisk && ramdisk_path) {
             std::cout << *(uint64_t *) ramdiskBuffer << "\n";
            // Kanav: This could break when the endianness of the machine changes
            osuCrypto::block b = osuCrypto::toBlock(*(uint64_t *) (ramdiskBuffer + 8), *(uint64_t *) ramdiskBuffer);
            ramdiskBuffer += sizeof(osuCrypto::block);
            bytesReceived += sizeof(osuCrypto::block);
            return b;
        }
        this->file.read(buf, sizeof(osuCrypto::block));
        //std::cout<< "dealer recv block" << "\n";
    } else {
       // recv(consocket, buf, sizeof(osuCrypto::block), MSG_WAITALL);
        std::cout << "dealer recv mask" << "\n";
    }
    osuCrypto::block b = *(osuCrypto::block *)buf;
    bytesReceived += sizeof(osuCrypto::block);
    return b;
}

GroupElement Dealer::recv_ge(int bl) {
    if (bl > 32) {
        char buf[8];
        if (useFile) {
            if (ramdisk && ramdisk_path) {
                GroupElement g = *(uint64_t *)ramdiskBuffer;
                ramdiskBuffer += 8;
                bytesReceived += 8;
                mod(g, bl);
                return g;
            }
            this->file.read(buf, 8);
            //std::cerr << "dealer recv ge 32" << "\n";
        } else {
           // recv(consocket, buf, 8, MSG_WAITALL);
            std::cout << "dealer recv mask" << "\n";
        }
        GroupElement g(*(uint64_t *)buf);
        mod(g, bl);
        bytesReceived += 8;
        return g;
    }
    else if (bl > 16) {
        char buf[4];
        if (useFile) {
            if (ramdisk && ramdisk_path) {
                GroupElement g = *(uint32_t *)ramdiskBuffer;
                ramdiskBuffer += 4;
                bytesReceived += 4;
                mod(g, bl);
                return g;
            }
            this->file.read(buf, 4);
            //std::cout << "dealer recv ge 16" << "\n";
        } else {
           // recv(consocket, buf, 4, MSG_WAITALL);
            std::cout << "dealer recv mask" << "\n";
        }
        GroupElement g(*(uint32_t *)buf);
        mod(g, bl);
        bytesReceived += 4;
        return g;
    }
    else if (bl > 8) {
        char buf[2];
        if (useFile) {
            if (ramdisk && ramdisk_path) {
                GroupElement g = *(uint16_t *)ramdiskBuffer;
                ramdiskBuffer += 2;
                bytesReceived += 2;
                mod(g, bl);
                return g;
            }
            this->file.read(buf, 2);
            //std::cout<< "dealer recv ge 8" << "\n";
        } else {
            //recv(consocket, buf, 2, MSG_WAITALL);
            std::cout << "dealer recv mask" << "\n";
        }
        GroupElement g(*(uint16_t *)buf);
        mod(g, bl);
        bytesReceived += 2;
        return g;
    }
    else {
        char buf[1];
        if (useFile) {
            if (ramdisk && ramdisk_path) {
                GroupElement g = *(uint8_t *)ramdiskBuffer;
                ramdiskBuffer += 1;
                bytesReceived += 1;
                mod(g, bl);
                return g;
            }
           this->file.read(buf, 1);
            //std::cout << "dealer recv ge 1" << "\n";
        } else {
           // recv(consocket, buf, 1, MSG_WAITALL);
            std::cout << "dealer recv mask" << "\n";
        }
        GroupElement g(*(uint8_t *)buf);
        mod(g, bl);
        bytesReceived += 1;
        return g;
    }
}


void Dealer::recv_ge_array(const GroupElement *g, int size) {
    char *buf = (char *)g;
    if (useFile) {
        if (ramdisk && ramdisk_path) {
            memcpy(buf, ramdiskBuffer, 8*size);
            ramdiskBuffer += 8*size;
            bytesReceived += 8*size;
            return;
        }
        this->file.read(buf, 8*size);
        //std::cout << "dealer recv ge array" << "\n";
    } else {
       // recv(consocket, buf, 8*size, MSG_WAITALL);
        std::cout << "dealer recv mask" << "\n";
    }
    bytesReceived += 8 * size;
    
}

DCFKeyPack Dealer::recv_dcf_keypack(int Bin, int Bout, int groupSize) {
    DCFKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.groupSize = groupSize;

    if (ramdisk && ramdisk_path) {
        kp.k = (osuCrypto::block *)ramdiskBuffer;
        ramdiskBuffer += sizeof(osuCrypto::block) * (Bin + 1);
    } else {
        kp.k = new osuCrypto::block[Bin + 1];
        for (int i = 0; i < Bin + 1; ++i) {
            kp.k[i] = recv_block();
        }
    }
    // kp.g = (GroupElement *)ramdiskBuffer;
    // ramdiskBuffer += sizeof(GroupElement) * groupSize;
    kp.g = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i) {
        kp.g[i] = recv_ge(Bout);
    }
    if (false) {
        kp.v = (GroupElement *)ramdiskBuffer;
        ramdiskBuffer += sizeof(GroupElement) * (Bin * groupSize);
    } else {
        kp.v = new GroupElement[Bin * groupSize];
        for (int i = 0; i < Bin * groupSize; ++i) {
            kp.v[i] = recv_ge(Bout);
        }
    }
    GroupElement t = recv_ge(64);
    always_assert(t == 42);
    return kp;
}

DualDCFKeyPack Dealer::recv_ddcf_keypack(int Bin, int Bout, int groupSize) {
    DualDCFKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.groupSize = groupSize;
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    kp.sb = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i) {
        kp.sb[i] = recv_ge(Bout);
    }
    return kp;
}

MultKeyNew Dealer::recv_new_mult_key(int bw1, int bw2)
{
    MultKeyNew kp;
    kp.a = recv_ge(bw1 + bw2);
    kp.b = recv_ge(bw1 + bw2);
    kp.c = recv_ge(bw1 + bw2);
    kp.k1 = recv_dcf_keypack(bw2, bw1, 1);
    kp.k2 = recv_dcf_keypack(bw1, bw2, 1);
    kp.k3 = recv_dcf_keypack(bw2, bw1, 1);
    kp.k4 = recv_dcf_keypack(bw1, bw2, 1);

    return kp;
}

MatMulKey Dealer::recv_matmul_key(int Bin, int Bout, int s1, int s2, int s3) {
    MatMulKey k;
    k.Bin = Bin;
    k.Bout = Bout;
    k.s1 = s1;
    k.s2 = s2;
    k.s3 = s3;

    k.a = make_array<GroupElement>(s1, s2);
    k.b = make_array<GroupElement>(s2, s3);
    k.c = make_array<GroupElement>(s1, s3);

    for(int i = 0; i < s1; ++i) {
        for(int j = 0; j < s2; ++j) {
            Arr2DIdx(k.a, s1, s2, i, j) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
            mod(Arr2DIdx(k.a, s1, s2, i, j), Bin);
        }
    }
    
    for(int i = 0; i < s2; ++i) {
        for(int j = 0; j < s3; ++j) {
            Arr2DIdx(k.b, s2, s3, i, j) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
            mod(Arr2DIdx(k.b, s2, s3, i, j), Bin);
        }
    }

    for(int i = 0; i < s1; ++i) {
        for(int j = 0; j < s3; ++j) {
            Arr2DIdx(k.c, s1, s3, i, j) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bout));
            mod(Arr2DIdx(k.c, s1, s3, i, j), Bout);
        }
    }

    return k;
}

Conv2DKey Dealer::recv_conv2d_key(int Bin, int Bout, int64_t N, int64_t H, int64_t W,
                int64_t CI, int64_t FH, int64_t FW,
                int64_t CO, int64_t zPadHLeft,
                int64_t zPadHRight, int64_t zPadWLeft,
                int64_t zPadWRight, int64_t strideH,
                int64_t strideW) {
    Conv2DKey k;
    k.Bin = Bin;
    k.Bout = Bout;
    k.N = N;
    k.H = H;
    k.W = W;
    k.CO = CO;
    k.CI = CI;
    k.FH = FH;
    k.FW = FW;
    k.zPadHLeft = zPadHLeft;
    k.zPadHRight = zPadHRight; 
    k.zPadWLeft = zPadWLeft;
    k.zPadWRight = zPadWRight;;
    k.strideH = strideH;
    k.strideW = strideW;

    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;

    k.a = make_array<GroupElement>(N, H, W, CI);
    k.b = make_array<GroupElement>(FH, FW, CI, CO);
    k.c = make_array<GroupElement>(d0, d1, d2, d3);

    for(int n = 0; n < N; ++n) {
        for(int h = 0; h < H; ++h) {
            for(int w = 0; w < W; ++w) {
                for(int ci = 0; ci < CI; ++ci) {
                    Arr4DIdx(k.a, N, H, W, CI, n, h, w, ci) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
                    mod(Arr4DIdx(k.a, N, H, W, CI, n, h, w, ci), Bin);
                }
            }
        }
    }

    for(int fh = 0; fh < FH; ++fh) {
        for(int fw = 0; fw < FW; ++fw) {
            for(int ci = 0; ci < CI; ++ci) {
                for(int co = 0; co < CO; ++co) {
                    Arr4DIdx(k.b, FH, FW, CI, CO, fh, fw, ci, co) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
                    mod(Arr4DIdx(k.b, FH, FW, CI, CO, fh, fw, ci, co), Bin);
                }
            }
        }
    }

    GroupElement *c = k.c;
    for(int i = 0; i < d0; ++i) {
        for(int j = 0; j < d1; ++j) {
            for(int k = 0; k < d2; ++k) {
                for(int l = 0; l < d3; ++l) {
                    Arr4DIdx(c, d0, d1, d2, d3, i, j, k, l) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bout));
                    mod(Arr4DIdx(c, d0, d1, d2, d3, i, j, k, l), Bout);
                }
            }
        }
    }
    return k;
}

Conv3DKey Dealer::recv_conv3d_key(int Bin, int Bout, int64_t N, int64_t D, int64_t H, int64_t W,
                   int64_t CI, int64_t FD, int64_t FH, int64_t FW, int64_t CO,
                   int64_t zPadDLeft, int64_t zPadDRight,
                   int64_t zPadHLeft, int64_t zPadHRight, 
                   int64_t zPadWLeft, int64_t zPadWRight, 
                   int64_t strideD, int64_t strideH, int64_t strideW) {
    Conv3DKey k;
    k.Bin = Bin;
    k.Bout = Bout;
    k.N = N;
    k.D = D;
    k.H = H;
    k.W = W;
    k.CO = CO;
    k.CI = CI;
    k.FD = FD;
    k.FH = FH;
    k.FW = FW;
    k.zPadDLeft = zPadDLeft;
    k.zPadDRight = zPadDRight;
    k.zPadHLeft = zPadHLeft;
    k.zPadHRight = zPadHRight; 
    k.zPadWLeft = zPadWLeft;
    k.zPadWRight = zPadWRight;
    k.strideD = strideD;
    k.strideH = strideH;
    k.strideW = strideW;

    int d0 = N;
    int d1 = ((D - FD + (zPadDLeft + zPadDRight)) / strideD) + 1;
    int d2 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d3 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d4 = CO;

    k.a = make_array<GroupElement>(N, D, H, W, CI);
    k.b = make_array<GroupElement>(FD, FH, FW, CI, CO);
    k.c = make_array<GroupElement>(d0, d1, d2, d3, d4);

    for(size_t i = 0; i < N * D * H * W * CI; ++i) {
        k.a[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
        mod(k.a[i], Bin);
    }

    for(size_t i = 0; i < FD * FH * FW * CI * CO; ++i) {
        k.b[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
        mod(k.b[i], Bin);
    }

    for(size_t i = 0; i < d0 * d1 * d2 * d3 * d4; ++i) {
        k.c[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bout));
        mod(k.c[i], Bout);
    }
    return k;
}

ReluKeyPack Dealer::recv_relu_key(int Bin, int Bout) {
    int groupSize = 2;
    ReluKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.g = new GroupElement[groupSize];
    // kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    if (ramdisk && ramdisk_path) {
        kp.k = (osuCrypto::block *)ramdiskBuffer;
        ramdiskBuffer += sizeof(osuCrypto::block) * (Bin + 1);
    } else {
        kp.k = new osuCrypto::block[Bin + 1];
        for(int i = 0; i < Bin + 1; ++i) {
            kp.k[i] = recv_block();
        }
    }
    for(int i = 0; i < groupSize; ++i) {
        kp.g[i] = recv_ge(Bout);
    }
    if (ramdisk && ramdisk_path && (Bin > 32)) {
        kp.v = (GroupElement *)ramdiskBuffer;
        ramdiskBuffer += sizeof(GroupElement) * (Bin * groupSize);
    }
    else {
        kp.v = new GroupElement[Bin * groupSize];
        for(int i = 0; i < Bin * groupSize; ++i) {
            kp.v[i] = recv_ge(Bout);
        }
    }
    kp.e_b0 = recv_ge(Bout);
    kp.e_b1 = recv_ge(Bout);
    kp.beta_b0 = recv_ge(Bout);
    kp.beta_b1 = recv_ge(Bout);
    kp.r_b = recv_ge(Bout);
    kp.drelu = recv_ge(1);
    return kp;
}

MaxpoolKeyPack Dealer::recv_maxpool_key(int Bin, int Bout) {
    MaxpoolKeyPack kp;
    kp.Bin = Bin; 
    kp.Bout = Bout;
    kp.reluKey = recv_relu_key(Bin, Bout);
    kp.rb = recv_ge(Bout);
    return kp;
}

MaxpoolDoubleKeyPack Dealer::recv_maxpool_double_key(int Bin, int Bout) {
    MaxpoolDoubleKeyPack kp;
    kp.Bin = Bin; 
    kp.Bout = Bout;
    kp.reluKey = recv_relu_2round_key(Bin, Bout);
    kp.rb = recv_ge(Bout);
    return kp;
}

ScmpKeyPack Dealer::recv_scmp_keypack(int Bin, int Bout) {
    int groupSize = 1;
    ScmpKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.dualDcfKey = recv_ddcf_keypack(Bin-1, Bout, groupSize);
    kp.rb = recv_ge(Bout);
    return kp;
}

PublicDivKeyPack Dealer::recv_pubdiv_key(int Bin, int Bout) {
    int groupSize = 1;
    PublicDivKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.dualDcfKey = recv_ddcf_keypack(Bin, Bout, groupSize);
    kp.scmpKey = recv_scmp_keypack(Bin, Bout);
    kp.zb = recv_ge(Bout);
    return kp;
}

ARSKeyPack Dealer::recv_ars_key(int Bin, int Bout, int shift) {
    ARSKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.shift = shift;

    int dcfGroupSize = 1, ddcfGroupSize = 2;
    if (!LlamaConfig::stochasticT)
        kp.dcfKey = recv_dcf_keypack(shift, Bout, dcfGroupSize);
    if (Bout > Bin - shift) {
        kp.dualDcfKey = recv_ddcf_keypack(Bin - 1, Bout, ddcfGroupSize);
    }
    kp.rb = recv_ge(Bout);
    return kp;
}

void Peer::sync() {
    char buf[1] = {1};
    send(sendsocket, buf, 1, 0);
    recv(recvsocket, buf, 1, MSG_WAITALL);
    bytesReceived += 1;
    bytesSent += 1;
    always_assert(buf[0] == 1);
}

SignedPublicDivKeyPack Dealer::recv_signedpubdiv_key(int Bin, int Bout) {
    SignedPublicDivKeyPack kp;
    kp.d = recv_ge(Bin);
    kp.Bin = Bin;
    kp.Bout = Bout;
    int groupSize = 1;
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    kp.publicICkey = recv_publicIC_key(Bin, Bout);
    kp.scmpKey = recv_scmp_keypack(Bin, Bout);
    kp.A_share = recv_ge(Bin);
    kp.corr_share = recv_ge(Bout);
    kp.B_share = recv_ge(Bout);
    kp.rdiv_share = recv_ge(Bout);
    kp.rout_temp_share = recv_ge(Bout);
    kp.rout_share = recv_ge(Bout);
    return kp;
}

PublicICKeyPack Dealer::recv_publicIC_key(int Bin, int Bout) {
    PublicICKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    int groupSize = 1;
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    kp.zb = recv_ge(Bout);
    return kp;
}


SplineKeyPack Dealer::recv_spline_key(int Bin, int Bout, int numPoly, int degree)
{
    SplineKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.numPoly = numPoly;
    kp.degree = degree;
    kp.dcfKey = recv_dcf_keypack(16, Bout, numPoly * (degree + 1));

    kp.p.resize(numPoly + 1);
    for(int i = 0; i < numPoly + 1; ++i) {
        kp.p[i] = recv_ge(Bin);
    }

    kp.e_b.resize(numPoly);
    for(int i = 0; i < numPoly; ++i) {
        kp.e_b[i].resize(degree + 1);
        for(int j = 0; j < degree + 1; ++j) {
            kp.e_b[i][j] = recv_ge(Bout);
        }
    }

    kp.beta_b.resize(numPoly * (degree + 1));
    for(int i = 0; i < numPoly * (degree + 1); ++i) {
        kp.beta_b[i] = recv_ge(Bout);
    }

    kp.r_b = recv_ge(Bout);
    return kp;
}

ReluTruncateKeyPack Dealer::recv_relu_truncate_key(int Bin, int Bout, int s) {
    ReluTruncateKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.shift = s;
    kp.dcfKeyN = recv_dcf_keypack(Bin, s, 1);
    if (!LlamaConfig::stochasticRT)
        kp.dcfKeyS = recv_dcf_keypack(s, Bin, 1);
    kp.zTruncate = recv_ge(Bin);
    kp.a = recv_ge(Bin);
    kp.b = recv_ge(Bin);
    kp.c = recv_ge(Bin);
    kp.d1 = recv_ge(Bin);
    kp.d2 = recv_ge(Bin);
    return kp;
}

Relu2RoundKeyPack Dealer::recv_relu_2round_key(int effectiveBin, int Bin)
{
    Relu2RoundKeyPack kp;
    kp.effectiveBin = effectiveBin;
    kp.Bin = Bin;
    kp.dcfKey = recv_dcf_keypack(effectiveBin, 1, 1);
    kp.a = recv_ge(Bin);
    kp.b = recv_ge(Bin);
    kp.c = recv_ge(Bin);
    kp.d1 = recv_ge(Bin);
    kp.d2 = recv_ge(Bin);
    return kp;
}

SelectKeyPack Dealer::recv_select_key(int Bin) {
    SelectKeyPack kp;
    kp.Bin = Bin;
    kp.a = recv_ge(Bin);
    kp.b = recv_ge(Bin);
    kp.c = recv_ge(Bin);
    kp.d1 = recv_ge(Bin);
    kp.d2 = recv_ge(Bin);
    return kp;
}

// void Peer::send_bulkylrs_key(const BulkyLRSKeyPack &kp, int bl, int m) {
//     send_dcf_keypack(kp.dcfKeyN);
//     for(int i = 0; i < m; ++i)
//     {
//         send_dcf_keypack(kp.dcfKeyS[i]);
//         send_ge(kp.z[i], bl);
//     }
//     send_ge(kp.out, bl);
// }

// void Peer::send_taylor_key(const TaylorKeyPack &kp, int bl, int m) {
//     send_dcf_keypack(kp.msnzbKey.micKey.dcfKey);
//     for(int i = 0; i < m; ++i)
//     {
//         send_ge(kp.msnzbKey.micKey.z[i], bl);
//     }
//     send_ge(kp.msnzbKey.r, bl);
//     send_ge(kp.squareKey.a, bl);
//     send_ge(kp.squareKey.b, bl);
//     send_bulkylrs_key(kp.lrsKeys[0], bl, m);
//     send_bulkylrs_key(kp.lrsKeys[1], bl, m);
//     send_bulkylrs_key(kp.lrsKeys[2], bl, m);
//     send_ge(kp.privateScaleKey.rin, bl);
//     send_ge(kp.privateScaleKey.rout, bl);
// }

BulkyLRSKeyPack Dealer::recv_bulkylrs_key(int bl, int m, uint64_t *scales) {
    BulkyLRSKeyPack kp;
    kp.dcfKeyN = recv_dcf_keypack(bl, bl, 1);
    kp.z = new GroupElement[m];
    kp.dcfKeyS = new DCFKeyPack[m];
    for(int i = 0; i < m; ++i)
    {
        kp.dcfKeyS[i] = recv_dcf_keypack(scales[i], bl, 1);
        kp.z[i] = recv_ge(bl);
    }
    kp.out = recv_ge(bl);
    return kp;
}

TaylorKeyPack Dealer::recv_taylor_key(int bl, int m, int sf) {
    TaylorKeyPack kp;
    kp.msnzbKey.micKey.dcfKey = recv_dcf_keypack(bl, bl, 1);
    kp.msnzbKey.micKey.z = new GroupElement[m];
    for(int i = 0; i < m; ++i)
    {
        kp.msnzbKey.micKey.z[i] = recv_ge(bl);
    }
    kp.msnzbKey.r = recv_ge(bl);
    kp.squareKey.a = recv_ge(bl);
    kp.squareKey.b = recv_ge(bl);
    kp.privateScaleKey.rin = recv_ge(bl);
    kp.privateScaleKey.rout = recv_ge(bl);
    uint64_t scales[m];
    for(int i = 0; i < m; ++i)
    {
        scales[i] = sf + i;
    }
    kp.lrsKeys[0] = recv_bulkylrs_key(bl, m, scales);
    for(int i = 0; i < m; ++i)
    {
        scales[i] = sf + 3 * i;
    }
    kp.lrsKeys[1] = recv_bulkylrs_key(bl, m, scales);
    GroupElement ping = recv_ge(bl);
    always_assert(ping == 69);
    return kp;
}

BitwiseAndKeyPack Dealer::recv_bitwise_and_key()
{
    BitwiseAndKeyPack kp;
    kp.t[0] = recv_ge(64);
    kp.t[1] = recv_ge(64);
    kp.t[2] = recv_ge(64);
    kp.t[3] = recv_ge(64);
    return kp;
}

// void Peer::send_mic_key(const MICKeyPack &kp, int bl, int m)
// {
//     send_dcf_keypack(kp.dcfKey);
//     for(int i = 0; i < m; ++i)
//     {
//         send_ge(kp.z[i], bl);
//     }
// }

// void Peer::send_fix_to_float_key(const FixToFloatKeyPack &kp, int bl)
// {
//     send_mic_key(kp.micKey, bl, 2*bl);
// }

MICKeyPack Dealer::recv_mic_key(int bin, int bout, int m)
{
    MICKeyPack kp;
    kp.dcfKey = recv_dcf_keypack(bin, bout, 1);
    kp.z = new GroupElement[m];
    for(int i = 0; i < m; ++i)
    {
        kp.z[i] = recv_ge(bout);
    }
    return kp;
}

FixToFloatKeyPack Dealer::recv_fix_to_float_key(int bl)
{
    FixToFloatKeyPack kp;
    kp.micKey = recv_mic_key(bl, bl, 2*bl);
    kp.rs = recv_ge(1);
    kp.rpow = recv_ge(bl);
    kp.ry = recv_ge(bl);
    kp.selectKey = recv_select_key(bl);
    kp.rm = recv_ge(bl);
    return kp;
}

FloatToFixKeyPack Dealer::recv_float_to_fix_key(int bl)
{
    FloatToFixKeyPack kp;
    kp.rm = recv_ge(24);
    kp.re = recv_ge(10);
    kp.dcfKey = recv_dcf_keypack(24, 1, 1);
    kp.rw = recv_ge(1);
    kp.rt = recv_ge(bl);
    kp.selectKey = recv_select_key(bl);
    for(int i = 0; i < 1024; ++i) {
        kp.p[i] = recv_ge(bl);
    }
    for(int i = 0; i < 1024; ++i) {
        kp.q[i] = recv_ge(bl);
    }
    return kp;
}

ReluExtendKeyPack Dealer::recv_relu_extend_key(int bin, int bout)
{
    ReluExtendKeyPack kp;
    kp.dcfKey = recv_dcf_keypack(bin, 2, 1);
    kp.rd = recv_ge(2);
    kp.rw = recv_ge(2);
    for(int i = 0; i < 4; ++i) {
        kp.p[i] = recv_ge(bout);
    }
    for(int i = 0; i < 2; ++i) {
        kp.q[i] = recv_ge(bout);
    }
    return kp;
}

SignExtend2KeyPack Dealer::recv_sign_extend2_key(int Bin, int Bout)
{
    SignExtend2KeyPack kp;
    kp.dcfKey = recv_dcf_keypack(Bin, 1, 1);
    kp.rw = recv_ge(1);
    kp.p[0] = recv_ge(Bout);
    kp.p[1] = recv_ge(Bout);
    return kp;
}

TripleKeyPack Dealer::recv_triple_key(int bw, int64_t na, int64_t nb, int64_t nc)
{
    TripleKeyPack k;
    k.bw = bw;
    k.na = na;
    k.nb = nb;
    k.nc = nc;
    k.a = make_array<GroupElement>(na);
    k.b = make_array<GroupElement>(nb);
    k.c = make_array<GroupElement>(nc);

    for(size_t i = 0; i < na; ++i) {
        k.a[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(bw));
        mod(k.a[i], bw);
    }

    for(size_t i = 0; i < nb; ++i) {
        k.b[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(bw));
        mod(k.b[i], bw);
    }

    for(size_t i = 0; i < nc; ++i) {
        k.c[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(bw));
        mod(k.c[i], bw);
    }
    return k;
}