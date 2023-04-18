#pragma once
#include <sytorch/tensor.h>
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

inline std::vector<float> take_input(size_t last)
{
    std::vector<float> _ret;
    for (size_t i = 0; i < last; i++)
    {
        std::cin >> _ret[i];
    }
    return _ret;
}

template <typename... Args>
auto take_input(size_t first, Args... sizes)
{
    auto _inner = take_input(sizes...);
    std::vector<decltype(_inner)> _ret;
    _ret.push_back(_inner);
    for (size_t i = 1; i < first; i++)
    {
        _ret.push_back(take_input(sizes...));
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
void blprint(const Tensor4D<T> &p, u64 bw)
{
    for (int i = 0; i < p.d1; ++i) {
        for (int j = 0; j < p.d2; ++j) {
            for (int k = 0; k < p.d3; ++k) {
                for (int l = 0; l < p.d4; ++l) {
                    i64 val;
                    if (bw == 64) {
                        val = p(i, j, k, l);
                    }
                    else {
                        val = (p(i, j, k, l) + (1LL << (bw - 1))) % (1LL << bw);
                        val -= (1LL << (bw - 1));
                    }
                    std::cout << val << " ";
                }
                if (p.d4 > 1) {
                    std::cout << std::endl;
                }
            }
            if (p.d3 > 1) {
                std::cout << std::endl;
            }
        }
        if (p.d2 > 1) {
            std::cout << std::endl;
        }
    }
    if (p.d1 > 1) {
        std::cout << std::endl;
    }
}

template <typename T>
void blprint(const Tensor4D<T> &p, u64 bw, u64 scale)
{
    for (int i = 0; i < p.d1; ++i) {
        for (int j = 0; j < p.d2; ++j) {
            for (int k = 0; k < p.d3; ++k) {
                for (int l = 0; l < p.d4; ++l) {
                    if (bw == 64) {
                        std::cout << ((double)p(i, j, k, l)) / (1LL << scale) << " ";
                        continue;
                    }
                    else {
                        i64 val = (p(i, j, k, l) + (1LL << (bw - 1))) % (1LL << bw);
                        val -= (1LL << (bw - 1));
                        std::cout << ((double)val) / (1LL << scale) << " ";
                    }
                }
                if (p.d4 > 1) {
                    std::cout << std::endl;
                }
            }
            if (p.d3 > 1) {
                std::cout << std::endl;
            }
        }
        if (p.d2 > 1) {
            std::cout << std::endl;
        }
    }
    if (p.d1 > 1) {
        std::cout << std::endl;
    }
}


template <typename T>
void print(const Tensor<T> &p, u64 bw = sizeof(T) * 8)
{
    u64 d = p.shape.back();
    for (u64 i = 0; i < p.size(); ++i)
    {
        i64 val;
        if (bw == sizeof(T) * 8) {
            val = p.data[i];
        }
        else {
            val = (p.data[i] + (1LL << (bw - 1))) % (1LL << bw);
            val -= (1LL << (bw - 1));
        }
        std::cout << val;
        if ((i + 1) % d == 0) {
            std::cout << std::endl;
        }
        else {
            std::cout << " ";
        }
    }
}

template <typename T>
Tensor2D<T> reshapeInputTransposed3d(const Tensor5D<T> &input, u64 padding, u64 stride, u64 FD, u64 FH, u64 FW) {
    u64 D = input.d2;
    u64 H = input.d3;
    u64 W = input.d4;
    u64 CI = input.d5;
    u64 newD = (((D + 2*padding - FD)/stride) + 1);
    u64 newH = (((H + 2*padding - FH)/stride) + 1);
	u64 newW = (((W + 2*padding - FW)/stride) + 1);
	u64 reshapedIPCols = input.d1 * newD * newH * newW;
    Tensor2D<T> reshaped(reshapedIPCols, FD * FH * FW * CI);
    i64 linIdxFilterMult = 0;
	for (i64 n = 0; n < input.d1; n++){
        i64 leftTopCornerD = 0 - padding;
        i64 extremeRightBottomCornerD = D - 1 + padding;
        while((leftTopCornerD + FD - 1) <= extremeRightBottomCornerD) {
            i64 leftTopCornerH = 0 - padding;
            i64 extremeRightBottomCornerH = H - 1 + padding;
            while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
                i64 leftTopCornerW = 0 - padding;
                i64 extremeRightBottomCornerW = W - 1 + padding;
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
                    leftTopCornerW = leftTopCornerW + stride;
                }
                leftTopCornerH = leftTopCornerH + stride;
            }
            leftTopCornerD = leftTopCornerD + stride;
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
