/*
Authors: Anwesh Bhattacharya
Copyright:
Copyright (c) 2021 Microsoft Research
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

#include "cleartext_library_float.h"
#include "math.h"

using namespace std ;

float intToFloat(int32_t m) {
	return (float)m ;
}

void Softmax1(int32_t s1,  vector<float>& inArr, vector<float>& outArr) {
	float sum = 0.0 ;
	float max = inArr[0] ;

	for (int i = 1 ; i < s1 ; i++)
		if (max > inArr[i])
			max = inArr[i] ;

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i] = exp(inArr[i]-max) ;
		sum += outArr[i] ;
	}	

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i] /= sum ;
	}
}

void Softmax2(int32_t s1, int32_t s2, vector<vector<float>>& inArr, vector<vector<float>>& outArr) {
	for (int i = 0 ; i < s1 ; i++)
		Softmax1(s2, inArr[i], outArr[i]) ;
}

void Ln(int32_t s1, vector<float>& inArr, vector<float>& outArr) {
	for (int i = 0 ; i < s1 ; i++) {
		outArr[i] = log(inArr[i]) ;
	}
}

void Tanh(int32_t s1, vector<float>& inArr, vector<float>& outArr) {
	for (int i = 0 ; i < s1 ; i++) {
		outArr[i] = inArr[i] + 1 ;
	}
}

void ElemWiseAdd(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) {
	for (int i = 0 ; i < s1 ; i++)
		outArr[i] = arr1[i] + arr2[i] ;
}

void ElemWiseSub(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) {
	for (int i = 0 ; i < s1 ; i++)
		outArr[i] = arr1[i] - arr2[i] ;
}

void ElemWiseMul(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) {
	for (int i = 0 ; i < s1 ; i++)
		outArr[i] = arr1[i] * arr2[i] ;
}

void ElemWiseDiv(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) {
	for (int i = 0 ; i < s1 ; i++)
		outArr[i] = arr1[i] / arr2[i] ;
}

void getOutDer(int32_t s1, int32_t s2, vector<vector<float>>& batchSoft, vector<vector<float>>& lab, vector<vector<float>>& der) {
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
der[i1][i2] = ((batchSoft[i1][i2] - lab[i1][i2]) / intToFloat(s1)) ;
}
}
}

void MatMul(int32_t s1, int32_t s2, int32_t s3, vector<vector<float>>& mat1, vector<vector<float>>& mat2, vector<vector<float>>& mat3) {
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i3 = 0; i3 < s3; i3++){
mat3[i1][i3] = 0. ;

for (uint32_t i2 = 0; i2 < s2; i2++){
mat3[i1][i3] = (mat3[i1][i3] + (mat1[i1][i2] * mat2[i2][i3])) ;

}
}
}
}

void GemmAdd(int32_t s1, int32_t s2, vector<vector<float>>& prod, vector<float>& bias, vector<vector<float>>& out) {
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
out[i1][i2] = (prod[i1][i2] + bias[i2]) ;

}
}
}

void dotProduct2(int32_t s1, int32_t s2, vector<vector<float>>& arr1, vector<vector<float>>& arr2, vector<float>& outArr) {
for (uint32_t i = 0; i < s1; i++){
outArr[i] = 0. ;

for (uint32_t j = 0; j < s2; j++){
outArr[i] = (outArr[i] + (arr1[i][j] * arr2[i][j])) ;

}
}
}

void vsumIfElse(int32_t s1, int32_t s2, vector<vector<float>>& arr, vector<vector<bool>>& hotArr, vector<float>& outArr) {
	for (uint32_t i = 0 ; i < s1 ; i++) {
		outArr[i] = 0.0 ;
		for (uint32_t j = 0 ; j < s2 ; j++) {
			outArr[i] = outArr[i] + (hotArr[i][j] ? arr[i][j] : 0.0) ;
		}
	}
}

void getLoss(int32_t m, vector<float>& lossTerms, vector<float>& loss) {
loss[0] = 0. ;

for (uint32_t i = 0; i < m; i++){
loss[0] = (loss[0] + lossTerms[i]) ;

}
loss[0] = ((0. - loss[0]) / intToFloat(m)) ;

}

void computeMSELoss(int32_t m, int32_t s, vector<vector<float>>& target, vector<vector<float>>& fwdOut, vector<float>& loss) {
loss[0] = 0. ;

float term ;

for (uint32_t i = 0; i < m; i++){
term = (fwdOut[i][0] - target[i][0]) ;

loss[0] = (loss[0] + (term * term)) ;

}
loss[0] = (loss[0] / (2. * intToFloat(m))) ;

}

void getBiasDer(int32_t s1, int32_t s2, vector<vector<float>>& der, vector<float>& biasDer) {
for (uint32_t i2 = 0; i2 < s2; i2++){
biasDer[i2] = der[0][i2] ;

for (uint32_t i1 = 1; i1 < s1; i1++){
biasDer[i2] = (biasDer[i2] + der[i1][i2]) ;

}
}
}

void updateWeightsAdam(
	int32_t sz, int32_t t, float lr, float beta1, float beta2, float eps, 
	vector<float>& wt, vector<float>& der, vector<float>& m_t, vector<float>& v_t
	) {
	float beta1_t = pow(beta1, t) ;
	float beta2_t = pow(beta2, t) ;

	vector<float> m_tcap = make_vector<float>(sz) ;
	vector<float> v_tcap = make_vector<float>(sz) ;

	for (uint32_t i = 0 ; i < sz ; i++) {
		m_t[i] = beta1*m_t[i] + der[i]*(1-beta1) ;
		v_t[i] = beta2*v_t[i] + der[i]*der[i]*(1-beta2) ;

		m_tcap[i] = m_t[i]/(1-beta1_t) ;
		v_tcap[i] = v_t[i]/(1-beta2_t) ;

		wt[i] = wt[i] - lr*m_tcap[i]/(eps + sqrt(v_tcap[i])) ;
	}
}


void updateWeightsMomentum(int32_t sz, float lr, float beta, vector<float>& inArr, vector<float> &derArr, vector<float> &momArr) {

	/*
		mom <- beta*mom +(1-beta)der
		weight <- weight - lr*mom
	*/

	for (uint32_t i = 0 ; i < sz ; i++) {
		momArr[i] = beta*momArr[i] + (1.0-beta)*derArr[i] ;
		inArr[i] = inArr[i] - lr*momArr[i] ;
	}
}

void updateWeights(int32_t s, float lr, vector<float>& bias, vector<float>& der) {
for (uint32_t i = 0; i < s; i++){
bias[i] = (bias[i] - (lr * der[i])) ;
}
}

void IfElse(int32_t s1, vector<float>& dat, vector<bool>& hot, vector<float>& out, bool flip) {
	for (uint32_t i1 = 0; i1 < s1; i1++) {
		if (flip)
			out[i1] = hot[i1] ? 0. : dat[i1] ;
		else
			out[i1] = hot[i1] ? dat[i1] : 0. ;
	}
}

void Relu(int32_t s1, vector<float>& inArr, vector<float>& outArr, vector<bool>& hotArr) {
	for (uint32_t i1 = 0; i1 < s1; i1++){
		hotArr[i1] = (inArr[i1] < 0.) ;
		outArr[i1] = hotArr[i1] ? 0.0 : inArr[i1] ;	
	}
}


void Conv2DReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, vector<vector<float>>& inputArr, vector<vector<vector<vector<float>>>>& outputArr){
	int32_t COG = (CO / G) ;
	int32_t startCO = (g * COG) ;
	for (uint32_t co = 0; co < COG; co++) {
		for (uint32_t n = 0; n < N; n++) {
			for (uint32_t h = 0; h < finalH; h++) {
				for (uint32_t w = 0; w < finalW; w++) {
					outputArr[n][h][w][(co + startCO)] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)] ;
				}
			}
		}
	}
}

void Conv2DReshapeFilterGroup(int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t g, int32_t G, vector<vector<vector<vector<float>>>>& inputArr, vector<vector<float>>& outputArr){
	int32_t CIG = (CI / G) ;
	int32_t COG = (CO / G) ;
	int32_t startCO = (g * COG) ;

	for (uint32_t co = 0; co < COG; co++) {
		for (uint32_t fh = 0; fh < FH; fh++) {
			for (uint32_t fw = 0; fw < FW; fw++) {
				for (uint32_t ci = 0; ci < CIG; ci++) {
					int32_t linIdx = ((((fh * FW) * CIG) + (fw * CIG)) + ci) ;
					outputArr[co][linIdx] = inputArr[fh][fw][ci][(co + startCO)] ;
				}
			}
		}
	}
}

void Conv2DReshapeInputGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t g, int32_t G, int32_t RRows, int32_t RCols, vector<vector<vector<vector<float>>>>& inputArr,  vector<vector<float>>& outputArr){
	int32_t linIdxFilterMult = 0 ;
	int32_t CIG = (CI / G) ;

	for (uint32_t n = 0; n < N; n++) {
		int32_t leftTopCornerH = (0 - zPadHLeft) ;
		int32_t extremeRightBottomCornerH = ((H - 1) + zPadHRight) ;

		while ((((leftTopCornerH + FH) - 1) <= extremeRightBottomCornerH)) {
			int32_t leftTopCornerW = (0 - zPadWLeft) ;
			int32_t extremeRightBottomCornerW = ((W - 1) + zPadWRight) ;

			while ((((leftTopCornerW + FW) - 1) <= extremeRightBottomCornerW)) {
				for (uint32_t fh = 0; fh < FH; fh++) {
					for (uint32_t fw = 0; fw < FW; fw++) {

						int32_t curPosH = (leftTopCornerH + fh) ;
						int32_t curPosW = (leftTopCornerW + fw) ;
						float val = 0.0 ;
						int32_t startCI = (g * CIG) ;

						for (uint32_t ci = 0; ci < CIG; ci++) {
							if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))) {
								val = 0.0 ;
							} else {
								val = inputArr[n][curPosH][curPosW][(ci + startCI)] ;
							}

							outputArr[((((fh * FW) * CIG) + (fw * CIG)) + ci)][linIdxFilterMult] = val ;
						}
					}
				}

				linIdxFilterMult = (linIdxFilterMult + 1) ;
				leftTopCornerW = (leftTopCornerW + strideW) ;
			}

			leftTopCornerH = (leftTopCornerH + strideH) ;
		}

	}
}

void Conv2DGroupWrapper(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, 
	int32_t strideH, int32_t strideW, int32_t G, 
	vector<vector<vector<vector<float>>>>& inputArr, 
	vector<vector<vector<vector<float>>>>& filterArr, 
	vector<vector<vector<vector<float>>>>& outArr) {

	int32_t CIG = (CI / G) ;
	int32_t reshapedFilterRows = (CO / G) ;
	int32_t reshapedFilterCols = ((FH * FW) * CIG) ;
	int32_t reshapedIPRows = ((FH * FW) * CIG) ;
	int32_t outH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + 1) ;
	int32_t outW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + 1) ;
	int32_t reshapedIPCols = ((N * outH) * outW) ;

	for (uint32_t g = 0; g < G; g++) {
		vector<vector<float>> inputReshaped = make_vector<float>(reshapedIPRows, reshapedIPCols) ;
		vector<vector<float>> filterReshaped = make_vector<float>(reshapedFilterRows, reshapedFilterCols) ;
		vector<vector<float>> matmulOP = make_vector<float>(reshapedFilterRows, reshapedIPCols) ;

		Conv2DReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
		Conv2DReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
		MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP);
		Conv2DReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);
	}
}


void ConvAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, 
	vector<vector<vector<vector<float>>>>& inArr, 
	vector<float>& biasArr, 
	vector<vector<vector<vector<float>>>>& outArr) {
	int sz ;
	sz = s1*s2*s3*s4 ;

	vector<float> arr1 = make_vector<float>(sz) ;
	vector<float> arr2 = make_vector<float>(sz) ;
	vector<float> out = make_vector<float>(sz) ;

	for (int i1=0 ; i1 < s1 ; i1++) {
		for (int i2 = 0 ; i2 < s2 ; i2++) {
			for (int i3 = 0 ; i3 < s3 ; i3++) {
				for (int i4 = 0 ; i4 < s4 ; i4++) {
					arr1[i1*s2*s3*s4 + i2*s3*s4 + i3*s4 + i4] = inArr[i1][i2][i3][i4] ;
					arr2[i1*s2*s3*s4 + i2*s3*s4 + i3*s4 + i4] = biasArr[i4] ;
				}
			}
		}
	}

	ElemWiseAdd(sz, arr1, arr2, out) ;

	for (int i1=0 ; i1 < s1 ; i1++) {
		for (int i2 = 0 ; i2 < s2 ; i2++) {
			for (int i3 = 0 ; i3 < s3 ; i3++) {
				for (int i4 = 0 ; i4 < s4 ; i4++) {
					outArr[i1][i2][i3][i4] = out[i1*s2*s3*s4 + i2*s3*s4 + i3*s4 + i4] ;
				}
			}
		}
	}
}

void MaxPool(
	int32_t N, int32_t imgH, int32_t imgW, int32_t C, 
	int32_t ksizeH, int32_t ksizeW, 
	int32_t strideH, int32_t strideW,
	int32_t H, int32_t W,
	vector<vector<vector<vector<float>>>>& inArr, 
	vector<vector<vector<vector<bool>>>> &poolmask, 
	vector<vector<vector<vector<float>>>>& outArr) {
	int size = N*H*C*W ;
	int filter_size = ksizeH*ksizeW; 

	for (int n = 0, size_k=0 ; n < N ; n++) {
		for (int c = 0 ; c < C ; c++) {
			for (int h = 0 ; h < H ; h++) {
				for (int w = 0 ; w < W ; w++, size_k++) {
					float max_val ;
					int max_h, max_w ;
					int img_h, img_w ;
					img_h = h*strideH ;
					img_w = w*strideW ;

					if (img_h < 0 || img_h >= imgH || img_w < 0 || img_w >= imgW) {
						max_val = 0.0 ;
						max_h = -1 ;
						max_w = -1 ;
					}
					else {
						max_val = inArr[n][img_h][img_w][c] ;
						max_h = 0 ;
						max_w = 0 ;
					}
						
					for (int kh = 0, filter_k = 0 ; kh < ksizeH ; kh++) {
						img_h = h*strideH + kh ; 

						for (int kw = 0 ; kw < ksizeW ; kw++, filter_k++) {
							img_w = w*strideW + kw ;
							float val ;
							int this_h, this_w ;

							if (img_h < 0 || img_h >= imgH || img_w < 0 || img_w >= imgW) {
								val = 0.0 ;			
								this_h = -1 ;
								this_w = -1 ;
							}	
							else {
								val = inArr[n][img_h][img_w][c] ;
								this_h = kh ;
								this_w = kw ;
							}
								
							if (val > max_val) {
								max_val = val ;
								max_h = this_h ;
								max_w = this_w ;
							}
						}
					}

					outArr[n][h][w][c] = max_val ;
					int mask_h = h*ksizeH + max_h ;		// multiply with ksize instead of stride
					int mask_w = w*ksizeW + max_w ;		// multiply with ksize instead of stride
					if (mask_h >= 0 && mask_w >= 0)
						poolmask[n][mask_h][mask_w][c] = true ;
					else
						poolmask[n][mask_h][mask_w][c] = false ;
				}
			}
		}
	}
}

void ConvDerReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, 
vector<vector<float>>& inputArr, 
vector<vector<vector<vector<float>>>>& outputArr){
	int32_t COG = (CO / G) ;
	int32_t startCO = (g * COG) ;
	for (uint32_t co = 0; co < COG; co++){
	for (uint32_t n = 0; n < N; n++){
	for (uint32_t h = 0; h < finalH; h++){
	for (uint32_t w = 0; w < finalW; w++){
	inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)] = outputArr[n][h][w][(co + startCO)] ;
	}
	}
	}
	}
}

void ConvDerReshapeInputGroup(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t g, int32_t G, int32_t RRows, int32_t RCols, 
	vector<vector<vector<vector<float>>>>& inputArr, 
	vector<vector<float>>& outputArr){
	int32_t linIdxFilterMult = 0 ;

	int32_t CIG = (CI / G) ;

	for (uint32_t n = 0; n < N; n++){
	int32_t leftTopCornerH = (0 - zPadHLeft) ;

	int32_t extremeRightBottomCornerH = ((H - 1) + zPadHRight) ;

	while ((((leftTopCornerH + FH) - 1) <= extremeRightBottomCornerH)) {
	int32_t leftTopCornerW = (0 - zPadWLeft) ;

	int32_t extremeRightBottomCornerW = ((W - 1) + zPadWRight) ;

	while ((((leftTopCornerW + FW) - 1) <= extremeRightBottomCornerW)) {
	for (uint32_t fh = 0; fh < FH; fh++){
	for (uint32_t fw = 0; fw < FW; fw++){
	int32_t curPosH = (leftTopCornerH + fh) ;

	int32_t curPosW = (leftTopCornerW + fw) ;

	float val = 0.0 ;

	int32_t startCI = (g * CIG) ;

	for (uint32_t ci = 0; ci < CIG; ci++){
	if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))) {
	val = 0.0 ;

	} else {
	val = inputArr[n][curPosH][curPosW][(ci + startCI)] ;

	}
	outputArr[linIdxFilterMult][((((fh * FW) * CIG) + (fw * CIG)) + ci)] = val ;

	}
	}
	}
	linIdxFilterMult = (linIdxFilterMult + 1) ;

	leftTopCornerW = (leftTopCornerW + strideW) ;

	}
	leftTopCornerH = (leftTopCornerH + strideH) ;
	}
	}
}

// inputArr needed 	--> [5, 5, 20, 50]
// we have 			--> [50, 20, 5, 5]
// loop 			--> [50, 5, 5, 20]
void ConvDerReshapeFilterGroup(int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t g, int32_t G, 
vector<vector<vector<vector<float>>>>& inputArr, 
vector<vector<float>>& outputArr){
	int32_t CIG = (CI / G) ;
	int32_t COG = (CO / G) ;
	int32_t startCO = (g * COG) ;

	for (uint32_t co = 0; co < COG; co++){
	for (uint32_t fh = 0; fh < FH; fh++){
	for (uint32_t fw = 0; fw < FW; fw++){
	for (uint32_t ci = 0; ci < CIG; ci++){
	int32_t linIdx = ((((fh * FW) * CIG) + (fw * CIG)) + ci) ;
	inputArr[fh][fw][ci][(co + startCO)] = outputArr[co][linIdx] ;
	}
	}
	}
	}
}


void ConvDerWrapper(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, 
	int32_t strideH, int32_t strideW, int32_t G, 
	vector<vector<vector<vector<float>>>>& inputArr, 
	vector<vector<vector<vector<float>>>>& filterArr, 
	vector<vector<vector<vector<float>>>>& outArr) {

	int32_t CIG = (CI / G) ;
	int32_t reshapedFilterRows = (CO / G) ;
	int32_t reshapedFilterCols = ((FH * FW) * CIG) ;
	int32_t reshapedIPRows = ((FH * FW) * CIG) ;
	int32_t outH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + 1) ;
	int32_t outW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + 1) ;
	int32_t reshapedIPCols = ((N * outH) * outW) ;

	for (uint32_t g = 0; g < G; g++) {
		vector<vector<float>> matmulOP = make_vector<float>(reshapedFilterRows, reshapedIPCols) ;
		ConvDerReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);

		vector<vector<float>> inputReshaped = make_vector<float>(reshapedIPCols, reshapedIPRows) ;
		ConvDerReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
		
		vector<vector<float>> filterReshaped = make_vector<float>(reshapedFilterRows, reshapedFilterCols) ;

		MatMul(reshapedFilterRows, reshapedIPCols, reshapedFilterCols, matmulOP, inputReshaped, filterReshaped);
		ConvDerReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
	}
}

void ConvBiasDer(
	int m, int W, int H, int chan, 
	vector<vector<vector<vector<float>>>> &der, 
	vector<float> &biasDer) {

	for (int c = 0 ; c < chan ; c++) {
		biasDer[c] = 0.0 ;

		for (int x = 0 ; x < m ; x++)
			for (int y = 0 ; y < W ; y++)
				for (int z = 0 ; z < H ; z++)
					biasDer[c] += der[x][y][z][c] ;
	}
}

vector<vector<vector<float>>> batched_matrix_multiplication(
	vector<vector<vector<float>>> &x, 
	vector<vector<vector<float>>> &y) {

	vector<vector<vector<float>>> ret ;
	int A = x.size() ;

	for (int i = 0 ; i < A ; i++) {
		int m, n, p ;
		m = x[i].size() ; n = x[i][0].size() ; p = y[i][0].size() ;
		vector<vector<float>> matout = make_vector<float>(m, p) ;
		MatMul(m, n, p, x[i], y[i], matout) ;
		ret.push_back(matout) ;
	}

	return ret ;
}

int num_mul_terms(
	int c, int i, int j,
	int inH, int inW,
	int filterH, int filterW,
	int outC
	) {
	int min_h, max_h, min_w, max_w ;

	min_h = max(i - filterH + 1, 0) ;
	max_h = min(i, inH - filterH) ;
	min_w = max(j - filterW + 1, 0) ;
	max_w = min(j, inW - filterW) ;

	return (max_h - min_h + 1)*(max_w - min_w + 1)*outC ;
}

void get_mul_terms(
	int BATCH,
	int inc, int inh, int inw,
	int inH, int inW,
	int filterH, int filterW,
	int outC,
	vector<vector<vector<vector<float>>>> &der, vector<vector<vector<vector<float>>>> &filter,
	map<int, vector<vector<vector<float>>>> &ld1, map<int, vector<vector<vector<float>>>> &ld2,
	map<tuple<int, int, int>, int> &tup_mp
	) {
	
	int min_h, max_h, min_w, max_w ;
	min_h = max(inh - filterH + 1, 0) ;
	max_h = min(inh, inH - filterH) ;
	min_w = max(inw - filterW + 1, 0) ;
	max_w = min(inw, inW - filterW) ;

	int terms = num_mul_terms(inc, inh, inw, inH, inW, filterH, filterW, outC) ;
	vector<vector<float>> term1 = make_vector<float>(BATCH, terms) ;
	vector<vector<float>> term2 = make_vector<float>(terms, 1) ;
	
	int sz1, sz2, sz3, sz4 ;
	sz1 = der.size() ; sz2 = der[0].size() ; sz3 = der[0][0].size() ; sz4 = der[0][0][0].size() ;
	int szz1, szz2, szz3, szz4 ;
	szz1 = filter.size() ; szz2 = filter[0].size() ; szz3 = filter[0][0].size() ; szz4 = filter[0][0][0].size() ;

	for (int oc = 0, k = 0 ; oc < outC ; oc++) {
		for (int h = min_h ; h < max_h + 1 ; h++) {
			for (int w = min_w ; w < max_w + 1 ; w++, k++) {				
				for (int m = 0 ; m < BATCH ; m++)
					term1[m][k] = der[m][h][w][oc] ;

				term2[k][0] = filter[inh-h][inw-w][inc][oc] ;
			}
		}
	}

	if (ld1.find(terms) != ld1.end()) {
		tup_mp[make_tuple(inc, inh, inw)] = (int)ld1[terms].size() ;
		ld1[terms].push_back(term1) ;
		ld2[terms].push_back(term2) ;
	} else {
		vector<vector<vector<float>>> fp1_vec, fp2_vec ;
		fp1_vec.push_back(term1) ;
		fp2_vec.push_back(term2) ;

		ld1[terms] = fp1_vec ;
		ld2[terms] = fp2_vec ;
		tup_mp[make_tuple(inc, inh, inw)] = 0 ;
	}
}

void GetPooledDer(
	int batch_size,
	int inH, int inW, int inC,
	int outC, int outH, int outW,
	int filterH, int filterW,
	vector<vector<vector<vector<float>>>> &conv_weights, 
	vector<vector<vector<vector<float>>>> &outDer, 
	vector<vector<vector<vector<float>>>> &inDer) {

	map<int, vector<vector<vector<float>>>> len_dot1 ;
	map<int, vector<vector<vector<float>>>> len_dot2 ;
	map<tuple<int, int, int>, int> ind_mp ;
	map<int, vector<vector<vector<float>>>> len_res ;

	for (int c = 0 ; c < inC ; c++) {
		for (int h = 0 ; h < inH ; h++) {
			for (int w = 0 ; w < inW ; w++) {
				get_mul_terms(batch_size, c, h, w, inH, inW, filterH, filterW, outC, outDer, conv_weights, len_dot1, len_dot2, ind_mp) ;
			}
		}
	}

	for (int i = 1 ; i <= filterH*filterW*outC ; i++)
		if (len_dot1.find(i) != len_dot1.end())
			len_res[i] = batched_matrix_multiplication(len_dot1[i], len_dot2[i]) ;

	for (int c = 0 ; c < inC ; c++) {
		for (int h = 0 ; h < inH ; h++) {
			for (int w = 0 ; w < inW ; w++) {
				int terms = num_mul_terms(c, h, w, inH, inW, filterH, filterW, outC) ;
				int ind = ind_mp[make_tuple(c, h, w)] ;

				int print1, print2 ;
				print1 = ind ;
				print2 = (int)len_dot1[terms].size() ;

				for (int m = 0 ; m < batch_size ; m++) {
					int div = len_res[terms][ind].size() ;
					inDer[m][h][w][c] = len_res[terms][ind][m/div][m%div] ;
				}
			}
		}
	}
}

/*
Given img1, pk, ps we have -

img2 = (img1 - pk)//ps
imgp = img2*pk

PooledDer --> [BATCH, img2, img2, outc]			This contains dL/do
Pool --> [BATCH, imgp, imgp, outc]				Hot map
ActDer --> [BATCH, img1, img1, outc]			Need to fill up dL/di
*/

void PoolProp(
	int32_t BATCH, int32_t outc, int32_t img2, int32_t imgp, int32_t img1, 
	int pk, int ps,
	vector<vector<vector<vector<float>>>> PooledDer, 
	vector<vector<vector<vector<bool>>>> Pool, 
	vector<vector<vector<vector<float>>>> ActDer, 
	bool flip) {
	map<pair<int, int>, int> inp_len ;					// cood -> len
	map<int, int> len_nchan ;			    			// len -> number of input indices
	map<pair<int, int>, int> inp_vind ;					// cood -> vector index

	map<int, vector<vector<float>>> len_derarr ;		// len -> derarr
	map<int, vector<vector<bool>>> len_condarr ;		// len -> condarr
	map<int, vector<float>> len_resarr ;				// len -> res

	/*
	1. fill in inp_len map. 
	2. fill in len_derarr, len_condarr map and execute len_resarrq for BATCH, ouc
	  : fill in inp_vind while iterating. Add index if no entry exists in inp_vind
	  : fill in len_nchan only when (b == 0 and c == 1). 
	Iterate over output image
	*/

	/*************** Building inp_len ***************/
	
	// Iterating over output image
	for (int oh = 0 ; oh < img2 ;  oh++) {
		for (int ow = 0 ; ow < img2 ; ow++) {

			// Iterate over pool indices
			for (int ph = 0 ; ph < pk ; ph++) {
				for (int pw = 0 ; pw < pk ; pw++) {
					int ih, iw ;
					ih = oh*ps + ph ;
					iw = ow*ps + pw ;
					pair<int, int> icood = make_pair(ih, iw) ;

					if (inp_len.find(icood) == inp_len.end()) {
						inp_len[icood] = 1 ;
					} else {
						inp_len[icood]++ ;
					}
				}
			}

		}
	}

	/*************** Building inp_vind and len_nchan ***************/

	// Iterating over input image
	for (int ih = 0 ; ih < (img2-1)*ps + pk ; ih++) {
		for (int iw = 0 ; iw < (img2-1)*ps + pk ; iw++) {
			pair<int, int> icood = make_pair(ih, iw) ;
			int len = inp_len[icood] ;

			if (len_nchan.find(len) == len_nchan.end()) {
				inp_vind[icood] = 0 ;
				len_nchan[len] = 1 ;
			} else {
				inp_vind[icood] = len_nchan[len] ;
				len_nchan[len]++ ;
			}
		}
	}

	/*************** Building len_derarr and len_condarr ***************/

	// Create empty len -> arr pairings

	vector<float> empty_farray ;
	vector<bool> empty_barray ;
	for (map<int,int>::iterator it = len_nchan.begin() ; it != len_nchan.end() ; it++) {
		int len = it->first ;
		int nchan = it->second ;

		vector<vector<float>> empty_derarr ;
		vector<vector<bool>> empty_condarr ;
		for (int i = 0 ; i < BATCH*outc*nchan ; i++) {
			empty_derarr.push_back(empty_farray) ;
			empty_condarr.push_back(empty_barray) ;
		}

		len_derarr[len] = empty_derarr ;
		len_condarr[len] = empty_condarr ;
	}

	// Iterating over batchsize, channels
	for (int b = 0 ; b < BATCH ; b++) {
		for (int c = 0 ; c < outc ; c++) {

			// Iterate over output image
			for (int oh = 0 ; oh < img2 ;  oh++) {
				for (int ow = 0 ; ow < img2 ; ow++) {

					// Iterate over pool indices
					for (int ph = 0 ; ph < pk ; ph++) {
						for (int pw = 0 ; pw < pk ; pw++) {
							pair<int, int> icood = make_pair(oh*ps + ph, ow*ps + pw) ;
							int len = inp_len[icood] ;
							int nchan = len_nchan[len] ;
							int vind = inp_vind[icood] ;

							len_derarr[len][b*outc*nchan + c*nchan + vind].push_back(PooledDer[b][oh][ow][c]) ;
							len_condarr[len][b*outc*nchan + c*nchan + vind].push_back(Pool[b][oh*pk+ph][ow*pk+pw][c]) ;
						}
					}

				}
			}

		}
	}

	for (map<int,int>::iterator it = len_nchan.begin() ; it != len_nchan.end() ; it++) {
		int len = it->first ;
		int nchan = it->second ;
		int s1, s2 ;
		s1 = BATCH*outc*nchan ;
		s2 = len ;

		len_resarr[len] = vector<float>(s1, 0.0) ;
		vsumIfElse(s1, s2, len_derarr[len], len_condarr[len], len_resarr[len]) ;
	}

	/*************** Computing len_resarr ***************/

	for (int b = 0 ; b < BATCH ; b++) {
		for (int c = 0 ; c < outc ; c++) {

			for (int ih = 0 ; ih < (img2-1)*ps + pk ; ih++) {
				for (int iw = 0 ; iw < (img2-1)*ps + pk ; iw++) {
					pair<int, int> icood = make_pair(ih, iw) ;
					int len = inp_len[icood] ;
					int nchan = len_nchan[len] ;

					ActDer[b][ih][iw][c] = len_resarr[len][b*outc*nchan + c*nchan + inp_vind[icood]] ;
				}
			}

		}
	}
}