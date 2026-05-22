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
#include <array>
// Array initializers
template <typename T> T *make_array(size_t s1) { return new T[s1]; }
template <typename T> T *make_array(size_t s1, size_t s2) {
    return new T[s1 * s2];
}
template <typename T> T *make_array(size_t s1, size_t s2, size_t s3) {
    return new T[s1 * s2 * s3];
}
template <typename T>
T *make_array(size_t s1, size_t s2, size_t s3, size_t s4) {
    return new T[s1 * s2 * s3 * s4];
}
template <typename T>
T *make_array(size_t s1, size_t s2, size_t s3, size_t s4, size_t s5) {
    return new T[s1 * s2 * s3 * s4 * s5];
}

// Indexing Helpers, we use 1D pointers for any dimension array, hence these macros are necessary
// Copied from SCI
#define Arr1DIdxRowM(arr, s0, i) (*((arr) + (i)))
#define Arr2DIdxRowM(arr, s0, s1, i, j) (*((arr) + (i) * (s1) + (j)))
#define Arr3DIdxRowM(arr, s0, s1, s2, i, j, k)                                 \
  (*((arr) + (i) * (s1) * (s2) + (j) * (s2) + (k)))
#define Arr4DIdxRowM(arr, s0, s1, s2, s3, i, j, k, l)                          \
  (*((arr) + (i) * (s1) * (s2) * (s3) + (j) * (s2) * (s3) + (k) * (s3) + (l)))
#define Arr5DIdxRowM(arr, s0, s1, s2, s3, s4, i, j, k, l, m)                   \
  (*((arr) + (i) * (s1) * (s2) * (s3) * (s4) + (j) * (s2) * (s3) * (s4) +      \
     (k) * (s3) * (s4) + (l) * (s4) + (m)))

