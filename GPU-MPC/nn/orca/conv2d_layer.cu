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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cmath>

#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_random.h"
#include "utils/helper_cuda.h"

#include "fss/gpu_conv2d.h"
#include "fss/dcf/gpu_truncate.h"
#include "fss/dcf/gpu_sgd.h"

#include "conv2d_layer.h"

namespace dcf
{
    namespace orca
    {
        template <typename T>
        Conv2DLayer<T>::Conv2DLayer(int bin, int bout, int N, int H, int W, int CI, int FH, int FW, int CO,
                                    int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, bool useBias, dcf::TruncateType tf, dcf::TruncateType tb, bool computedI, bool inputIsShares)
        {
            assert(bin == bout && bin <= sizeof(T) * 8);
            this->name = "Conv2D";
            p = {bin, bout, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, 0, 0, 0, 0, 0};
            fillConv2DParams(&p);

            this->useBias = useBias;
            this->s.comm_time = 0;
            this->s.transfer_time = 0;
            this->tf = tf;
            this->tb = tb;
            size_t memSizeI = p.size_I * sizeof(T);
            size_t memSizeF = p.size_F * sizeof(T);

            I = (T *)cpuMalloc(memSizeI);
            F = (T *)cpuMalloc(memSizeF);
            Vf = (T *)cpuMalloc(memSizeF);
            memset(F, 0, memSizeF);
            memset(Vf, 0, memSizeF);
            // this is okay to do because we will never use I and mask_I
            // at the same time
            mask_I = I;
            d_mask_I = NULL;
            mask_F = F;
            mask_Vf = Vf;

            if (useBias)
            {
                size_t memSizeB = p.CO * sizeof(T);
                b = (T *)cpuMalloc(memSizeB);
                Vb = (T *)cpuMalloc(memSizeB);
                memset(b, 0, memSizeB);
                memset(Vb, 0, memSizeB);
            }
            else
            {
                b = NULL;
                Vb = NULL;
            }
            mask_b = b;
            mask_Vb = Vb;
            this->computedI = computedI;
            this->inputIsShares = inputIsShares;

            initConvKey();
            initConvKeydI();
            initConvKeydF();
        }

        template <typename T>
        void Conv2DLayer<T>::initConvKey()
        {
            memcpy(&convKey, &p, sizeof(Conv2DParams));
            convKey.mem_size_I = p.size_I * sizeof(T);
            convKey.mem_size_F = p.size_F * sizeof(T);
            convKey.mem_size_O = p.size_O * sizeof(T);
        }

        template <typename T>
        void Conv2DLayer<T>::initConvKeydI()
        {
            memcpy(&convKeydI, &p, sizeof(Conv2DParams));
            convKeydI.p.size_I = p.size_O;
            convKeydI.p.size_F = p.size_F;
            convKeydI.p.size_O = p.size_I;
            convKeydI.mem_size_I = convKey.mem_size_O;
            convKeydI.mem_size_F = convKey.mem_size_F;
            convKeydI.mem_size_O = convKey.mem_size_I;
        }

        template <typename T>
        void Conv2DLayer<T>::initConvKeydF()
        {
            memcpy(&convKeydF, &p, sizeof(Conv2DParams));
            convKeydF.p.size_I = p.size_O;
            convKeydF.p.size_F = p.size_I;
            convKeydF.p.size_O = p.size_F;
            convKeydF.mem_size_I = convKey.mem_size_O;
            convKeydF.mem_size_F = convKey.mem_size_I;
            convKeydF.mem_size_O = convKey.mem_size_F;
        }

        template <typename T>
        T *Conv2DLayer<T>::genForwardKey(u8 **key_as_bytes, int party, T *d_mask_I, AESGlobalContext *gaes)
        {
            if (this->train)
                moveIntoCPUMem((u8 *)this->mask_I, (u8 *)d_mask_I, convKey.mem_size_I, NULL);
            auto d_mask_C = gpuKeygenConv2D<T>(key_as_bytes, party, convKey, d_mask_I, mask_F, true);
            // bias has scale 2s
            if (useBias)
                gpuAddBias(1, p.size_O / p.CO, p.CO, p.bout, d_mask_C, mask_b, NULL);

            auto d_mask_truncated_C = genGPUTruncateKey<T>(key_as_bytes, party, tf, p.bin, p.bout, global::scale, p.size_O, d_mask_C, gaes); /*, mask_truncated_C);*/
            // we don't need to free this because truncate does
            // gpuFree(d_mask_C);
            return d_mask_truncated_C;
        }

        // need to truncate dI
        template <typename T>
        T *Conv2DLayer<T>::genBackwardKey(u8 **key_as_bytes, int party, /*T**/ T *d_mask_grad, AESGlobalContext *gaes, int epoch)
        { // T* mask_truncated_dI) {
            this->checkIfTrain();
            // need to free all the leaked memory
            auto d_mask_dF = randomGEOnGpu<T>(p.size_F, p.bin);
            auto d_mask_I = (T *)moveToGPU((u8 *)mask_I, convKey.mem_size_I, NULL);
            auto d_masked_dF = gpuConv2DPlaintext<T>(convKeydF, d_mask_grad, d_mask_I, d_mask_dF, 2, false);

            writeShares<T, T>(key_as_bytes, party, p.size_O, d_mask_grad, p.bout);
            writeShares<T, T>(key_as_bytes, party, p.size_F, d_masked_dF, p.bout);

            gpuFree(d_masked_dF);
            gpuFree(d_mask_I);
            auto d_mask_F = (T *)moveToGPU((u8 *)mask_F, convKey.mem_size_F, NULL);

            // this needs to be computed here because ow the contents of mask_F will change
            T *d_mask_truncated_dI = NULL;
            if (computedI)
            {
                auto d_mask_dI = randomGEOnGpu<T>(p.size_I, p.bin);
                auto d_masked_dI = gpuConv2DPlaintext<T>(convKeydI, d_mask_grad, d_mask_F, d_mask_dI, 1, false);
                writeShares<T, T>(key_as_bytes, party, p.size_I, d_masked_dI, p.bout);
                gpuFree(d_masked_dI);
                d_mask_truncated_dI = genGPUTruncateKey<T>(key_as_bytes, party, tb, p.bin, p.bout, global::scale, p.size_I, d_mask_dI, gaes);
            }

            genOptimizerKey<T>(key_as_bytes, party, p.bin, p.bout, p.size_F, mask_F, d_mask_F, mask_Vf, d_mask_dF, global::scale, 2 * global::scale, 2 * global::scale, tb, this->useMomentum, gaes, epoch);

            if (useBias)
            {
                auto d_mask_db = getBiasGrad<T>(p.size_O / p.CO, p.CO, p.bin, d_mask_grad);
                // printf("Old Mask b=%ld, %ld, %ld\n", b[0], b[1], b[2]);
                genOptimizerKey<T>(key_as_bytes, party, p.bin, p.bout, p.CO, mask_b, NULL, mask_Vb, d_mask_db, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, tb, this->useMomentum, gaes, epoch);
                // printf("New Mask b=%ld, %ld, %ld\n", b[0], b[1], b[2]);
                gpuFree(d_mask_db);
            }

            gpuFree(d_mask_dF);
            gpuFree(d_mask_F);
            gpuFree(d_mask_grad);
            return d_mask_truncated_dI;
        }

        template <typename T>
        void Conv2DLayer<T>::readForwardKey(u8 **key_as_bytes)
        {
            initConvKey();
            convKey.I = (T *)*key_as_bytes;
            *key_as_bytes += convKey.mem_size_I;
            convKey.F = (T *)*key_as_bytes;
            *key_as_bytes += convKey.mem_size_F;
            convKey.O = (T *)*key_as_bytes;
            *key_as_bytes += convKey.mem_size_O;
            truncateKeyC = readGPUTruncateKey<T>(tf, key_as_bytes);
        }

        template <typename T>
        void Conv2DLayer<T>::readBackwardKey(u8 **key_as_bytes, int epoch)
        {
            T *mask_grad = (T *)*key_as_bytes;
            *key_as_bytes += convKey.mem_size_O;
            T *mask_dF = (T *)*key_as_bytes;
            *key_as_bytes += convKey.mem_size_F;

            // grad * input
            convKeydF.I = mask_grad;
            convKeydF.F = convKey.I;
            convKeydF.O = mask_dF;

            if (computedI)
            {
                T *mask_dI = (T *)*key_as_bytes;
                *key_as_bytes += convKey.mem_size_I;
                // grad * F
                convKeydI.I = mask_grad;
                convKeydI.F = convKey.F;
                convKeydI.O = mask_dI;

                // should refactor this later to look pretty
                truncateKeydI = readGPUTruncateKey<T>(tb, key_as_bytes);
            }
            // readGpuSGDWithMomentumKey(tb, &truncateKeyVf, &truncateKeyF, &truncateKeyVb, key_as_bytes, useBias);
            readOptimizerKey(tb, &truncateKeyVf, &truncateKeyF, key_as_bytes, global::scale, 2 * global::scale, 2 * global::scale, this->useMomentum, epoch);
            if (useBias)
                readOptimizerKey(tb, &truncateKeyVb, &truncateKeyb, key_as_bytes, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, this->useMomentum, epoch);
        }

        template <typename T>
        T *Conv2DLayer<T>::forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes)
        {
            T *d_F, *d_mask_F;
            d_mask_I = (T *)moveToGPU((u8 *)convKey.I, convKey.mem_size_I, &(this->s));
            if (inputIsShares)
            {
                gpuLinearComb(p.bin, p.size_I, d_I, T(1), d_I, T(1), d_mask_I);
                peer->reconstructInPlace(d_I, p.bin, p.size_I, &(this->s));
            }
            if (this->train)
                moveIntoCPUMem((u8 *)I, (u8 *)d_I, convKey.mem_size_I, &(this->s));

            d_F = (T *)moveToGPU((u8 *)F, convKey.mem_size_F, &(this->s));
            d_mask_F = (T *)moveToGPU((u8 *)convKey.F, convKey.mem_size_F, &(this->s));
            auto d_C = gpuConv2DBeaver(convKey, party, d_I, d_F, d_mask_I, d_mask_F, useBias && party == SERVER0 ? b : (T*) NULL, &(this->s), 0);

            // should not be freeing d_I who knows where else it is being used
            // gpuFree(d_I);
            gpuFree(d_F);
            gpuFree(d_mask_I);
            gpuFree(d_mask_F);

            peer->reconstructInPlace(d_C, p.bout, p.size_O, &(this->s));
            dcf::gpuTruncate(p.bin, p.bout, tf, truncateKeyC, global::scale, peer, party, p.size_O, d_C, gaes, &(this->s));

            return d_C;
        }

        template <typename T>
        T *Conv2DLayer<T>::backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *gaes, int epoch)
        {
            this->checkIfTrain();

            auto d_mask_incomingGrad = (T *)moveToGPU((u8 *)convKeydF.I, convKeydF.mem_size_I, &(this->s));
            auto d_mask_I = (T *)moveToGPU((u8 *)convKey.I, convKey.mem_size_I, &(this->s));
            auto d_I = (T *)moveToGPU((u8 *)I, convKey.mem_size_I, &(this->s));
            auto d_F = (T *)moveToGPU((u8 *)F, convKey.mem_size_F, &(this->s));

            T *d_dI = NULL;
            if (computedI)
            {
                auto d_mask_F = (T *)moveToGPU((u8 *)convKey.F, convKey.mem_size_F, &(this->s));
                d_dI = gpuConv2DBeaver(convKeydI, party, d_incomingGrad, d_F, d_mask_incomingGrad, d_mask_F, (T *)NULL, &(this->s), 1);
                gpuFree(d_mask_F);
                peer->reconstructInPlace(d_dI, p.bin, p.size_I, &(this->s));
                dcf::gpuTruncate(p.bin, p.bout, tb, truncateKeydI, global::scale, peer, party, p.size_I, d_dI, gaes, &(this->s));
            }

            auto d_dF = gpuConv2DBeaver(convKeydF, party, d_incomingGrad, d_I, d_mask_incomingGrad, d_mask_I, (T *)NULL, &(this->s), 2);
            peer->reconstructInPlace(d_dF, p.bin, p.size_F, &(this->s));

            optimize(p.bin, p.bout, p.size_F, F, d_F, Vf, d_dF, global::scale, 2 * global::scale, 2 * global::scale, tb, truncateKeyVf, truncateKeyF, party, peer, this->useMomentum, gaes, &(this->s), epoch);

            if (useBias)
            {
                auto d_db = getBiasGrad<T>(p.size_O / p.CO, p.CO, p.bout, d_incomingGrad);
                optimize(p.bin, p.bout, p.CO, b, (T *)NULL, Vb, d_db, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, tb, truncateKeyVb, truncateKeyb,
                              party, peer, this->useMomentum, gaes, &(this->s), epoch);
                gpuFree(d_db);
            }

            gpuFree(d_incomingGrad);
            gpuFree(d_I);
            gpuFree(d_F);
            gpuFree(d_mask_incomingGrad);
            gpuFree(d_mask_I);
            gpuFree(d_dF);

            return d_dI;
        }

        template <typename T>
        void Conv2DLayer<T>::initWeights(u8 **weights, bool floatWeights)
        {
            if (floatWeights)
            {
                for (int i = 0; i < p.size_F; i++)
                    F[i] = T(((float *)*weights)[i] * (1ULL << global::scale));
                // printf("F[%d]=%lu\n", p.size_F - 1, F[p.size_F - 1]);
                *weights += (p.size_F * sizeof(float));
                if (useBias)
                {
                    for (int i = 0; i < p.CO; i++)
                        b[i] = T(((float *)*weights)[i] * (1ULL << (2 * global::scale)));
                    *weights += (p.CO * sizeof(float));
                }
            }
            else
            {
                memcpy(F, *weights, p.size_F * sizeof(T));
                *weights += (p.size_F * sizeof(T));
                if (useBias)
                {
                    memcpy(b, *weights, p.CO * sizeof(T));
                    *weights += (p.CO * sizeof(T));
                }
            }
        }

        template <typename T>
        void Conv2DLayer<T>::dumpWeights(std::ofstream &f)
        {
            f.write((char *)F, p.size_F * sizeof(T));
            // printf("Dumping weights=%lu, %lu, %lu\n", F[0], F[1], F[2]);
            if (useBias)
                f.write((char *)b, p.CO * sizeof(T));
        }
    }
}
