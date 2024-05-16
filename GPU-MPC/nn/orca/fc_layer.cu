// Author: Neha Jawalkar
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

#include "fc_layer.h"

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cmath>

#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_random.h"

#include "fss/gpu_matmul.h"
#include "fss/dcf/gpu_truncate.h"
#include "fss/dcf/gpu_sgd.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        FCLayer<T>::FCLayer(int bin, int bout, int M, int N, int K, dcf::TruncateType tf, dcf::TruncateType tb, bool useBias, bool computedX, bool inputIsShares)
        {
            assert(bin == bout && bin <= sizeof(T) * 8);
            assert(useBias);

            this->name = "FC";
            p.batchSz = 1;
            p.M = M;
            p.N = N;
            p.K = K;
            stdInit(p, bin, 0);
            initMulParamsdW();
            initMulParamsdX();
            initMemSz(p, &mmKey);
            initMemSz(pdW, &mmKeydW);
            initMemSz(pdX, &mmKeydX);

            size_t memSzX = p.size_A * sizeof(T);
            size_t memSzW = p.size_B * sizeof(T);
            size_t memSzZ = p.size_C * sizeof(T);

            mask_X = (T *)cpuMalloc(memSzX);
            mask_W = (T *)cpuMalloc(memSzW);
            mask_Z = (T *)cpuMalloc(memSzZ);
            mask_dX = (T *)cpuMalloc(memSzX);
            mask_dW = (T *)cpuMalloc(memSzW);
            mask_Vw = (T *)cpuMalloc(memSzW);

            memset(mask_W, 0, memSzW);
            memset(mask_Vw, 0, memSzW);

            this->tf = tf;
            this->tb = tb;
            this->useBias = useBias;
            this->computedX = computedX;
            this->inputIsShares = inputIsShares;
            if (useBias)
            {
                size_t memSzY = p.N * sizeof(T);
                mask_Y = (T *)cpuMalloc(memSzY);
                mask_Vy = (T *)cpuMalloc(memSzY);
                mask_dY = (T *)cpuMalloc(memSzY);
                memset(mask_Y, 0, memSzY);
                memset(mask_Vy, 0, memSzY);
            }
            X = mask_X;
            W = mask_W;
            Y = mask_Y;
            Vw = mask_Vw;
            Vy = mask_Vy;
        }

        template <typename T>
        void FCLayer<T>::initMulParamsdW()
        {
            pdW.batchSz = 1;
            pdW.M = p.K;
            pdW.K = p.M;
            pdW.N = p.N;
            stdInit(pdW, p.bw, 0);
            pdW.rowMaj_A = false;
        }

        template <typename T>
        void FCLayer<T>::initMulParamsdX()
        {
            pdX.batchSz = 1;
            pdX.M = p.M;
            pdX.K = p.N;
            pdX.N = p.K;
            stdInit(pdX, p.bw, 0);
            pdX.rowMaj_B = false;
        }

        // neha: to fix: maxpool, and make it so the conv2d output is 40 bits???? (bout == 40????)
        template <typename T>
        T *FCLayer<T>::genForwardKey(u8 **key_as_bytes, int party, T *d_mask_X, AESGlobalContext *gaes)
        {
            if (this->train)
                moveIntoCPUMem((u8 *)this->mask_X, (u8 *)d_mask_X, mmKey.mem_size_A, NULL);
            auto d_mask_Z = randomGEOnGpu<T>(p.size_C, p.bw);
            auto d_mask_W = (T *)moveToGPU((u8 *)mask_W, mmKey.mem_size_B, NULL);
            auto d_masked_Z = gpuMatmulPlaintext(p, d_mask_X, d_mask_W, d_mask_Z, false); //, true, true, true);
            writeShares<T, T>(key_as_bytes, party, p.size_A, d_mask_X, p.bw);
            writeShares<T, T>(key_as_bytes, party, p.size_B, d_mask_W, p.bw);
            writeShares<T, T>(key_as_bytes, party, p.size_C, d_masked_Z, p.bw);
            // gpuFree(d_mask_X);
            gpuFree(d_mask_W);
            gpuFree(d_masked_Z);
            if (useBias)
                gpuAddBias(1, p.M, p.N, p.bw, d_mask_Z, mask_Y, NULL);
            auto d_mask_truncated_Z = genGPUTruncateKey(key_as_bytes, party, tf, p.bw, p.bw, global::scale, p.size_C, d_mask_Z, gaes); /*, mask_truncated_C);*/
            // don't need to free this, it happens inside truncate
            // gpuFree(d_mask_Z);
            return d_mask_truncated_Z;
        }

        template <typename T>
        T *FCLayer<T>::genBackwardKey(u8 **key_as_bytes, int party, T *d_mask_grad, AESGlobalContext *gaes, int epoch)
        {
            this->checkIfTrain();

            auto d_mask_dW = randomGEOnGpu<T>(p.size_B, p.bw);
            auto d_mask_X = (T *)moveToGPU((u8 *)mask_X, mmKey.mem_size_A, NULL);
            auto d_masked_dW = gpuMatmulPlaintext(pdW, d_mask_X, d_mask_grad, d_mask_dW, false);

            writeShares<T, T>(key_as_bytes, party, p.size_C, d_mask_grad, p.bw);
            writeShares<T, T>(key_as_bytes, party, p.size_B, d_masked_dW, p.bw);
            gpuFree(d_masked_dW);
            gpuFree(d_mask_X);

            auto d_mask_W = (T *)moveToGPU((u8 *)mask_W, mmKey.mem_size_B, NULL);

            T *d_mask_truncated_dX = NULL;

            if (computedX)
            {
                auto d_mask_dX = randomGEOnGpu<T>(p.size_A, p.bw);
                auto d_masked_dX = gpuMatmulPlaintext(pdX, d_mask_grad, d_mask_W, d_mask_dX, false);
                writeShares<T, T>(key_as_bytes, party, p.size_A, d_masked_dX, p.bw);
                gpuFree(d_masked_dX);
                // d_mask_dX gets freed inside keygen for truncate
                d_mask_truncated_dX = genGPUTruncateKey(key_as_bytes, party, tb, p.bw, p.bw, global::scale, p.size_A, d_mask_dX, gaes);
            }
            genOptimizerKey(key_as_bytes, party, p.bw, p.bw, p.size_B, mask_W, d_mask_W, mask_Vw, d_mask_dW, global::scale, 2 * global::scale, 2 * global::scale, tb, this->useMomentum, gaes, epoch);
            if (useBias)
            {
                auto d_mask_dY = getBiasGrad(p.M, p.N, p.bw, d_mask_grad);
                genOptimizerKey(key_as_bytes, party, p.bw, p.bw, p.N, mask_Y, (T *)NULL, mask_Vy, d_mask_dY, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, tb, this->useMomentum, gaes, epoch);
                gpuFree(d_mask_dY);
            }
            gpuFree(d_mask_W);
            gpuFree(d_mask_dW);
            gpuFree(d_mask_grad);
            return d_mask_truncated_dX;
        }

        template <typename T>
        void FCLayer<T>::readForwardKey(u8 **key_as_bytes)
        {
            mmKey.A = (T *)*key_as_bytes;
            *key_as_bytes += mmKey.mem_size_A;

            mmKey.B = (T *)*key_as_bytes;
            *key_as_bytes += mmKey.mem_size_B;

            mmKey.C = (T *)*key_as_bytes;
            *key_as_bytes += mmKey.mem_size_C;

            truncateKeyZ = readGPUTruncateKey<T>(tf, key_as_bytes);
        }

        template <typename T>
        void FCLayer<T>::readBackwardKey(u8 **key_as_bytes, int epoch)
        {
            T *mask_grad = (T *)*key_as_bytes;
            *key_as_bytes += mmKey.mem_size_C;

            T *mask_dW = (T *)*key_as_bytes;
            *key_as_bytes += mmKey.mem_size_B;

            mmKeydW.A = mmKey.A;
            mmKeydW.B = mask_grad;
            mmKeydW.C = mask_dW;

            if (computedX)
            {
                T *mask_dX = (T *)*key_as_bytes;
                *key_as_bytes += mmKey.mem_size_A;

                mmKeydX.A = mask_grad;
                mmKeydX.B = mmKey.B;
                mmKeydX.C = mask_dX;

                truncateKeydX = readGPUTruncateKey<T>(tb, key_as_bytes);
            }

            readOptimizerKey(tb, &truncateKeyVw, &truncateKeyW, key_as_bytes, global::scale, 2 * global::scale, 2 * global::scale, this->useMomentum, epoch);
            if (useBias)
                readOptimizerKey(tb, &truncateKeyVy, &truncateKeyY, key_as_bytes, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, this->useMomentum, epoch);
        }

        template <typename T>
        T *FCLayer<T>::forward(SigmaPeer *peer, int party, T *d_X, AESGlobalContext *gaes)
        {
            auto d_mask_X = (T *)moveToGPU((u8 *)mmKey.A, mmKey.mem_size_A, &(this->s));
            if (inputIsShares)
            {
                gpuLinearComb(p.bw, p.size_A, d_X, T(1), d_X, T(1), d_mask_X);
                peer->reconstructInPlace(d_X, p.bw, p.size_A, &(this->s));
            }
            if (this->train)
                moveIntoCPUMem((u8 *)X, (u8 *)d_X, mmKey.mem_size_A, &(this->s));

            auto d_W = (T *)moveToGPU((uint8_t *)W, mmKey.mem_size_B, &(this->s));
            auto d_mask_W = (T *)moveToGPU((uint8_t *)mmKey.B, mmKey.mem_size_B, &(this->s));
            T *d_Y = nullptr;
            if (party == SERVER0 && useBias)
            {
                d_Y = (T *)moveToGPU((uint8_t *)Y, p.N * sizeof(T), &(this->s));
            }
            auto d_Z = gpuMatmulBeaver(p, mmKey, party, d_X, d_W, d_mask_X, d_mask_W, d_Y, &(this->s));

            // gpuFree(d_X);
            gpuFree(d_mask_X);
            gpuFree(d_W);
            gpuFree(d_mask_W);
            if (d_Y)
                gpuFree(d_Y);

            peer->reconstructInPlace(d_Z, p.bw, p.size_C, &(this->s));
            dcf::gpuTruncate(p.bw, p.bw, tf, truncateKeyZ, global::scale, peer, party, p.size_C, d_Z, gaes, &(this->s));

            return d_Z;
        }

        template <typename T>
        T *FCLayer<T>::backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *gaes, int epoch)
        {

            this->checkIfTrain();
            auto d_mask_grad = (T *)moveToGPU((u8 *)mmKeydW.B, mmKeydW.mem_size_B, &(this->s));
            auto d_X = (T *)moveToGPU((u8 *)X, mmKeydW.mem_size_A, &(this->s));
            auto d_mask_X = (T *)moveToGPU((u8 *)mmKeydW.A, mmKeydW.mem_size_A, &(this->s));
            auto d_W = (T *)moveToGPU((u8 *)W, mmKey.mem_size_B, &(this->s));

            T *d_dX;
            if (computedX)
            {
                auto d_mask_W = (T *)moveToGPU((u8 *)mmKeydX.B, mmKeydX.mem_size_B, &(this->s));
                d_dX = gpuMatmulBeaver(pdX, mmKeydX, party, d_incomingGrad, d_W, d_mask_grad, d_mask_W, (T *)NULL, &(this->s));
                gpuFree(d_mask_W);
                peer->reconstructInPlace(d_dX, p.bw, p.size_A, &(this->s));
                dcf::gpuTruncate(p.bw, p.bw, tb, truncateKeydX, global::scale, peer, party, p.size_A, d_dX, gaes, &(this->s));
            }

            auto d_dW = gpuMatmulBeaver(pdW, mmKeydW, party, d_X, d_incomingGrad, d_mask_X, d_mask_grad, (T *)NULL, &(this->s));
            peer->reconstructInPlace(d_dW, p.bw, p.size_B, &(this->s));

            gpuFree(d_X);
            gpuFree(d_mask_X);

            optimize(p.bw, p.bw, p.size_B, W, d_W, Vw, d_dW, global::scale, 2 * global::scale, 2 * global::scale, tb, truncateKeyVw, truncateKeyW, party, peer, this->useMomentum, gaes, &(this->s), epoch);

            gpuFree(d_W);
            gpuFree(d_dW);
            if (useBias)
            {
                auto d_dY = getBiasGrad(p.M, p.N, p.bw, d_incomingGrad);
                optimize(p.bw, p.bw, p.N, Y, (T *)NULL, Vy, d_dY, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, tb, truncateKeyVy, truncateKeyY,
                         party, peer, this->useMomentum, gaes, &(this->s), epoch);
                gpuFree(d_dY);
            }
            gpuFree(d_incomingGrad);
            gpuFree(d_mask_grad);

            return d_dX;
        }

        template <typename T>
        void FCLayer<T>::initWeights(u8 **weights, bool floatWeights)
        {
            if (floatWeights)
            {
                for (int i = 0; i < p.size_B; i++)
                    W[i] = T(((float *)*weights)[i] * (1ULL << global::scale));
                *weights += (p.size_B * sizeof(float));
                if (useBias)
                {
                    for (int i = 0; i < p.N; i++)
                        Y[i] = T(((float *)*weights)[i] * (1ULL << (2 * global::scale)));
                    *weights += (p.N * sizeof(float));
                }
            }
            else
            {
                size_t memSzW = p.size_B * sizeof(T);
                memcpy(W, *weights, memSzW);
                *weights += memSzW;
                if (useBias)
                {
                    memcpy(Y, *weights, p.N * sizeof(T));
                    *weights += (p.N * sizeof(T));
                }
            }
        }

        template <typename T>
        void FCLayer<T>::dumpWeights(std::ofstream &f)
        {
            f.write((char *)W, p.size_B * sizeof(T));
            if (useBias)
                f.write((char *)Y, p.N * sizeof(T));
        }

    }
}
