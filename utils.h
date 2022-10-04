#include "tensor.h"

template <typename T>
Tensor2D<T> matmul(const Tensor2D<T> &a, const Tensor2D<T> &b) {
    assert(a.d2 == b.d1);
    Tensor2D<T> c(a.d1, b.d2);
    c.zero();
    for(int i = 0; i < a.d1; i++) {
        for(int j = 0; j < b.d2; j++) {
            for(int k = 0; k < a.d2; k++) {
                c(i,j) += a(i, k) * b(k, j);
            }
        }
    }
    return c;
}

template <typename T>
Tensor4D<T> matmul(const Tensor4D<T> &a, const Tensor2D<T> &b) {
    assert(a.d2 == b.d1);
    assert(a.d3 == 1);
    assert(a.d4 == 1);
    Tensor4D<T> c(a.d1, b.d2, 1, 1);
    c.zero();
    for(int i = 0; i < a.d1; i++) {
        for(int j = 0; j < b.d2; j++) {
            for(int k = 0; k < a.d2; k++) {
                c(i, j, 0, 0) += a(i, k, 0, 0) * b(k, j);
            }
        }
    }
    return c;
}

template <typename T>
Tensor2D<T> matmul(const Tensor4D<T> &a, const Tensor4D<T> &b) {
    assert(a.d2 == b.d1);
    assert(a.d3 == 1);
    assert(a.d4 == 1);
    assert(b.d3 == 1);
    assert(b.d4 == 1);
    Tensor2D<T> c(a.d1, b.d2);
    c.zero();
    for(int i = 0; i < a.d1; i++) {
        for(int j = 0; j < b.d2; j++) {
            for(int k = 0; k < a.d2; k++) {
                c(i, j) += a(i, k, 0, 0) * b(k, j, 0, 0);
            }
        }
    }
    return c;
}

template <typename T>
Tensor4D<T> matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b) {
    assert(a.d2 == b.d2);
    assert(a.d3 == 1);
    assert(a.d4 == 1);
    Tensor4D<T> c(a.d1, b.d1, 1, 1);
    c.zero();
    for(int i = 0; i < a.d1; i++) {
        for(int j = 0; j < b.d1; j++) {
            for(int k = 0; k < a.d2; k++) {
                c(i, j, 0, 0) += a(i, k, 0, 0) * b(j, k);
            }
        }
    }
    return c;
}

template <typename T>
Tensor2D<T> reshapeFilter(const Tensor4D<T> &filter) {
    Tensor2D<T> res(filter.d4, filter.d1 * filter.d2 * filter.d3);
    for(int i = 0; i < filter.d4; i++) {
        for(int j = 0; j < filter.d1; j++) {
            for(int k = 0; k < filter.d2; k++) {
                for(int l = 0; l < filter.d3; l++) {
                    res(i, j * filter.d2 * filter.d3 + k * filter.d3 + l) = filter(j, k, l, i);
                }
            }
        }
    }
    return res;
}

template <typename T>
Tensor2D<T> reshapeInput(const Tensor4D<T> &input, u64 padding, u64 stride, u64 FH, u64 FW) {
    u64 newH = (((input.d2 + 2*padding - FH)/stride) + 1);
	u64 newW = (((input.d3 + 2*padding - FW)/stride) + 1);
	u64 reshapedIPCols = input.d1 * newH * newW;
    Tensor2D<T> reshaped(FH * FW * input.d4, reshapedIPCols);
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
								reshaped((fh*FW*input.d4) + (fw*input.d4) + ci, linIdxFilterMult) = 0L;
							}
							else{
								reshaped((fh*FW*input.d4) + (fw*input.d4) + ci, linIdxFilterMult) = input(n, curPosH, curPosW, ci);
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
Tensor4D<T> reshapeOutput(const Tensor2D<T> &output, u64 d1, u64 d2, u64 d3, u64 d4) {
    Tensor4D<T> res(d1, d2, d3, d4);
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
    return res;
}

template <typename T>
Tensor4D<T> conv2D(u64 padding, u64 stride, const Tensor4D<T> &input, const Tensor4D<T> &filter)
{
    Tensor2D<T> f = reshapeFilter<T>(filter);
    Tensor2D<T> i = reshapeInput<T>(input, padding, stride, filter.d1, filter.d2);
    Tensor2D<T> o = matmul<T>(f, i);
    return reshapeOutput<T>(o, input.d1, (((input.d2 + 2*padding - filter.d1)/stride) + 1), (((input.d3 + 2*padding - filter.d2)/stride) + 1), filter.d4);
}
