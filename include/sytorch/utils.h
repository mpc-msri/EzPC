#pragma once
#include <sytorch/tensor.h>
#include <queue>
#include <set>
#include <fstream>

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
void print_dot_graph(LayerTreeNode<T> *root)
{
    std::ofstream dotfile("graph.dot");
    dotfile << "digraph G {" << std::endl;
    std::queue<LayerTreeNode<T> *> q;
    std::set<LayerTreeNode<T> *> visited;
    q.push(root);
    while (!q.empty()) {
        LayerTreeNode<T> *curr = q.front();
        if (visited.find(curr) != visited.end()) {
            q.pop();
            continue;
        }
        q.pop();
        if (curr->curr != nullptr) {
            dotfile << curr->curr->name + std::to_string((u64)(curr->curr)) << " [label=\"" << curr->curr->name << "\"];" << std::endl;
        }
        for (auto &child : curr->children) {
            if (curr->curr != nullptr) {
                dotfile << curr->curr->name + std::to_string((u64)(curr->curr)) << " -> " << child->curr->name + std::to_string((u64)(child->curr)) << ";" << std::endl;
            }
            q.push(child);
        }
        visited.insert(curr);
    }
    dotfile << "}" << std::endl;
    dotfile.close();
}
