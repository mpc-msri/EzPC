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

#pragma once
#include <sytorch/tensor.h>
#include <llama/array.h>
#include <queue>
#include <set>
#include <fstream>

inline std::vector<float> make_float_vector( size_t last)
{
	std::vector<float> _ret;
	for (size_t i = 0; i < last; i++)
	{
		_ret.push_back(0.0);
	}
	return _ret;
}

template <typename... Args>
auto make_float_vector( size_t first, Args... sizes)
{
	auto _inner = make_float_vector( sizes...);
	std::vector<decltype(_inner)> _ret;
	_ret.push_back(_inner);
	for (size_t i = 1; i < first; i++)
	{
		_ret.push_back(make_float_vector( sizes...));
	}
	return _ret;
}

template <typename T>
Tensor2D<T> reshapeInputTransposed(const Tensor4D<T> &input, u64 padding, u64 stride, u64 FH, u64 FW) {
    u64 newH = (((input.d2 + 2*padding - FH)/stride) + 1);
	u64 newW = (((input.d3 + 2*padding - FW)/stride) + 1);
	u64 reshapedIPCols = input.d1 * newH * newW;
    Tensor2D<T> reshaped(reshapedIPCols, FH * FW * input.d4);
    i64 linIdxFilterMult = 0;
	for (i64 n = 0; n < input.d1; n++){
		i64 leftTopCornerH = 0 - padding;
		i64 extremeRightBottomCornerH = input.d2 - 1 + padding;
		while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
			i64 leftTopCornerW = 0 - padding;
			i64 extremeRightBottomCornerW = input.d3 - 1 + padding;
			while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

				for (i64 fh = 0; fh < FH; fh++){
					for (i64 fw = 0; fw < FW; fw++){
						i64 curPosH = leftTopCornerH + fh;
						i64 curPosW = leftTopCornerW + fw;
						for (i64 ci = 0; ci < input.d4; ci++){
							if ((((curPosH < 0) || (curPosH >= input.d2)) || ((curPosW < 0) || (curPosW >= input.d3)))){
								reshaped(linIdxFilterMult, (fh*FW*input.d4) + (fw*input.d4) + ci) = 0L;
							}
							else{
								reshaped(linIdxFilterMult, (fh*FW*input.d4) + (fw*input.d4) + ci) = input(n, curPosH, curPosW, ci);
							}
						}
					}
				}

				linIdxFilterMult = linIdxFilterMult + 1;
				leftTopCornerW = leftTopCornerW + stride;
			}

			leftTopCornerH = leftTopCornerH + stride;
		}
	}
    return reshaped;
}

template <typename T>
void reshapeOutput(const Tensor2D<T> &output, u64 d1, u64 d2, u64 d3, u64 d4, Tensor4D<T> &res) {
    assert(res.d1 == d1);
    assert(res.d2 == d2);
    assert(res.d3 == d3);
    assert(res.d4 == d4);
    assert(output.d1 == d4);
    assert(output.d2 == d1 * d2 * d3);
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                for(int l = 0; l < d4; l++) {
                    res(i, j, k, l) = output(l, i * d2 * d3 + j * d3 + k);
                }
            }
        }
    }
}

template <typename T>
void reshapeOutputReversed(Tensor2D<T> &output, u64 d1, u64 d2, u64 d3, u64 d4, const Tensor4D<T> &res) {
    assert(res.d1 == d1);
    assert(res.d2 == d2);
    assert(res.d3 == d3);
    assert(res.d4 == d4);
    assert(output.d1 == d4);
    assert(output.d2 == d1 * d2 * d3);
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                for(int l = 0; l < d4; l++) {
                    output(l, i * d2 * d3 + j * d3 + k) = res(i, j, k, l);
                }
            }
        }
    }
}

template <typename T>
void transposeFilter(u64 fh, u64 fw, u64 ci, u64 co, const Tensor2D<T> &filter, Tensor2D<T> &transposedFilter)
{
    assert(filter.d1 == co);
    assert(filter.d2 == fh * fw * ci);
    assert(transposedFilter.d1 == ci);
    assert(transposedFilter.d2 == fh * fw * co);

    for(int i = 0; i < fh; i++) {
        for(int j = 0; j < fw; j++) {
            for(int k = 0; k < ci; k++) {
                for(int l = 0; l < co; l++) {
                    transposedFilter(k, i * fw * co + j * co + l) = filter(l, (fh - i - 1) * fw * ci + (fw - j - 1) * ci + k);
                }
            }
        }
    }
}

template <typename T>
Tensor2D<T> reshapeInputTransposed3d(const Tensor5D<T> &input, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 FD, u64 FH, u64 FW) {
    u64 D = input.d2;
    u64 H = input.d3;
    u64 W = input.d4;
    u64 CI = input.d5;
    u64 newD = (((D + 2*pd - FD)/sd) + 1);
    u64 newH = (((H + 2*ph - FH)/sh) + 1);
	u64 newW = (((W + 2*pw - FW)/sw) + 1);
	u64 reshapedIPCols = input.d1 * newD * newH * newW;
    Tensor2D<T> reshaped(reshapedIPCols, FD * FH * FW * CI);
    i64 linIdxFilterMult = 0;
	for (i64 n = 0; n < input.d1; n++){
        i64 leftTopCornerD = 0 - pd;
        i64 extremeRightBottomCornerD = D - 1 + pd;
        while((leftTopCornerD + FD - 1) <= extremeRightBottomCornerD) {
            i64 leftTopCornerH = 0 - ph;
            i64 extremeRightBottomCornerH = H - 1 + ph;
            while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
                i64 leftTopCornerW = 0 - pw;
                i64 extremeRightBottomCornerW = W - 1 + pw;
                while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

                    for (i64 fd = 0; fd < FD; fd++) {
                        for (i64 fh = 0; fh < FH; fh++){
                            for (i64 fw = 0; fw < FW; fw++){
                                i64 curPosD = leftTopCornerD + fd;
                                i64 curPosH = leftTopCornerH + fh;
                                i64 curPosW = leftTopCornerW + fw;
                                for (i64 ci = 0; ci < CI; ci++){
                                    if ((((curPosD < 0) || (curPosD >= D)) || ((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))){
                                        reshaped(linIdxFilterMult, fd*(FH*FW*CI) + (fh*FW*CI) + (fw*CI) + ci) = 0L;
                                    }
                                    else{
                                        reshaped(linIdxFilterMult, fd*(FH*FW*CI) + (fh*FW*CI) + (fw*CI) + ci) = input(n, curPosD, curPosH, curPosW, ci);
                                    }
                                }
                            }
                        }
                    }

                    linIdxFilterMult = linIdxFilterMult + 1;
                    leftTopCornerW = leftTopCornerW + sw;
                }
                leftTopCornerH = leftTopCornerH + sh;
            }
            leftTopCornerD = leftTopCornerD + sd;
		}
	}
    return reshaped;
}

template <typename T>
void reshapeOutput3d(const Tensor2D<T> &output, u64 d1, u64 d2, u64 d3, u64 d4, u64 d5, Tensor5D<T> &res) {
    assert(res.d1 == d1);
    assert(res.d2 == d2);
    assert(res.d3 == d3);
    assert(res.d4 == d4);
    assert(res.d5 == d5);
    assert(output.d1 == d5);
    assert(output.d2 == d1 * d2 * d3 * d4);
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                for(int l = 0; l < d4; l++) {
                    for(int m = 0; m < d5; m++) {
                        res(i, j, k, l, m) = output(m, i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l);
                    }
                }
            }
        }
    }
}

template <typename T>
void convTranspose3dLoop(
    int64_t N, 
    int64_t D, 
    int64_t H, 
    int64_t W, 
    int64_t CI, 
    int64_t FD, 
    int64_t FH, 
    int64_t FW, 
    int64_t CO, 
    int64_t zPadDLeft, 
    int64_t zPadDRight, 
    int64_t zPadHLeft, 
    int64_t zPadHRight, 
    int64_t zPadWLeft, 
    int64_t zPadWRight, 
    int64_t strideD, 
    int64_t strideH, 
    int64_t strideW, 
    int64_t outD, 
    int64_t outH, 
    int64_t outW, 
    T* inputArr, 
    T* filterArr, 
    T* outArr)
{
    zPadDLeft = FD - 1 - zPadDLeft;
    zPadDRight = FD - 1 - zPadDRight;
    zPadHLeft = FH - 1 - zPadHLeft;
    zPadHRight = FH - 1 - zPadHRight;
    zPadWLeft = FW - 1 - zPadWLeft;
    zPadWRight = FW - 1 - zPadWRight;

    #pragma omp parallel for collapse(5)
    for (int64_t n =  0; n < N; n++){
        for (int64_t d =  0; d < outD; d++){
            for (int64_t h =  0; h < outH; h++){
                for (int64_t w =  0; w < outW; w++){
                    for (int64_t co =  0; co < CO; co++){
                        
                        T val =  0;
                        for (int64_t ci =  0; ci < CI; ci++){
                            for (int64_t fd = d; fd < (d + FD); fd++){
                                for (int64_t fh = h; fh < (h + FH); fh++){
                                    for (int64_t fw = w; fw < (w + FW); fw++){

                                        int64_t curPosD = ((fd - zPadDLeft) / strideD);
                                        int64_t curPosH = ((fh - zPadHLeft) / strideH);
                                        int64_t curPosW = ((fw - zPadWLeft) / strideW);

                                        if ((curPosD >=  0) &&
                                            (curPosH >=  0) &&
                                            (curPosW >=  0) &&
                                            (curPosD < D) &&
                                            (curPosH < H) &&
                                            (curPosW < W) &&
                                            (((fd - zPadDLeft) % strideD) == 0) &&
                                            (((fh - zPadHLeft) % strideH) == 0) &&
                                            (((fw - zPadWLeft) % strideW) == 0))
                                        {
                                            int32_t curFilterPosD = FD + d - fd -  1;
                                            int32_t curFilterPosH = FH + h - fh -  1;
                                            int32_t curFilterPosW = FW + w - fw -  1;
                                            val += (Arr5DIdx(inputArr, N, D, H, W, CI, n, curPosD, curPosH, curPosW, ci) * Arr5DIdx(filterArr, CO, FD, FH, FW, CI, co, curFilterPosD, curFilterPosH, curFilterPosW, ci));
                                        }
                                    }
                                }
                            }
                        }
                        Arr5DIdx(outArr, N, outD, outH, outW, CO, n, d, h, w, co) =  val;
                        // std::cout << "setting element at (" << n << " " << d << " " << h << " " << w << " " << co << ")" << std::endl;
                    }
                }
            }
        }
    }
}

template <typename T, typename... Args>
std::vector<T *> collect(T &first, Args & ... args)
{
    std::vector<T *> res;
    res.push_back(&first);
    collectHelper(res, args...);
    return res;
}
template <typename T, typename... Args>
void collectHelper(std::vector<T *> &res)
{

}

template <typename T, typename... Args>
void collectHelper(std::vector<T *> &res, T &a, Args & ... args)
{
    res.push_back(&a);
    collectHelper(res, args...);
}

template <typename T, typename... Args>
void collectByValueHelper(std::vector<T> &res)
{

}

template <typename T, typename... Args>
void collectByValueHelper(std::vector<T> &res, T a, Args ... args)
{
    res.push_back(a);
    collectByValueHelper(res, args...);
}

template <typename T, typename... Args>
std::vector<T> collectByValue(T first, Args ... args)
{
    std::vector<T> res;
    res.push_back(first);
    collectByValueHelper(res, args...);
    return res;
}

template <typename... Args>
std::string paramstring(Args ... args)
{
    std::stringstream ss;
    auto arr = collectByValue(args...);
    for (u64 i = 0; i < arr.size(); ++i)
    {
        ss << std::to_string(arr[i]) << "|";
    }
    return ss.str();
}

inline std::string paramstring()
{
    return "";
}

template <typename T>
std::vector<std::vector<u64>> getShapes(const std::vector<Tensor<T> *> &tensors) {
    std::vector<std::vector<u64>> shapes;
    for (auto tensor : tensors) {
        shapes.push_back(tensor->shape);
    }
    return shapes;
}

template <typename T>
void print(const Tensor<T> &p, u64 scale, u64 bw)
{
    printf("bw=%lu, scale=%lu, p[0]=%ld, p[1]=%ld\n", bw, scale, p.data[0], p.data[1]);
    u64 d = p.shape.back();
    int count = 0;
    for (u64 i = 0; i < /*p.size()*/10; ++i)
    {
        i64 val;
        if (bw == sizeof(T) * 8) {
            val = p.data[i];
        }
        else {
            val = (p.data[i] + (1LL << (bw - 1))) % (1LL << bw);
            val -= (1LL << (bw - 1));
        }
        std::cout << (double) val / (1LL << scale);
        if(std::abs((double) val / (1LL << scale)) < 1) count++; 
        if ((i + 1) % d == 0) {
            std::cout << std::endl;
        }
        else {
            std::cout << " ";
        }
    }
    // printf("Count < 1: %d\n", count);
}

template <typename T>
void print(const Tensor<T> &p, u64 scale)
{
    print(p, scale, sizeof(T) * 8);
}

inline void printshape(const std::vector<u64> &shape) {
    std::cout << "(";
    for(int i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << ", ";
    }
    std::cout << ")" << std::endl;
}

inline void sytorch_init()
{
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
}

template <typename T>
void qkv_split(Tensor2D<T> &x, Tensor4D<T> &y, u64 n_heads)
{
    always_assert(x.d2 % 3 == 0);
    u64 n_seq = x.d1;
    u64 n_embd = x.d2 / 3;
    always_assert(n_embd % n_heads == 0);
    always_assert(y.d1 == 3);
    always_assert(y.d2 == n_heads);
    always_assert(y.d3 == n_seq);
    always_assert(y.d4 == n_embd / n_heads);

    for (u64 i = 0; i < n_seq; ++i)
    {
        for (u64 j = 0; j < n_embd; ++j)
        {
            u64 head = j / (n_embd / n_heads);
            u64 pos = j % (n_embd / n_heads);
            y(0, head, i, pos) = x(i, j);
            y(1, head, i, pos) = x(i, j + n_embd);
            y(2, head, i, pos) = x(i, j + 2 * n_embd);
        }
    }
}
