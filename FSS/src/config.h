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

/* The below comments use (bitlen, scale) format to describe the functionality. */

/* To use Sigmoid and Tanh in (37, 12) uncomment this flag and comment all other SIGMOID and TANH flags */
// #define SIGMOID_TANH_37

/* To use Sigmoid in (bitlen, scale), uncomment `SIGMOID_bitlen_scale` flag and comment other SIGMOID flags and SIGMOID_TANH_37 flag. */
// #define SIGMOID_8_14
#define SIGMOID_9_14
// #define SIGMOID_11_14
// #define SIGMOID_13_14
// #define SIGMOID_12_12

/* To use Tanh in (bitlen, scale), uncomment `TANH_bitlen_scale` flag and comment other TANH flags and SIGMOID_TANH_37 flag. */
// #define TANH_8_8
#define TANH_9_9
// #define TANH_11_11
// #define TANH_12_12
// #define TANH_13_13

/* To use InvSqrt in (bitlen, scale), uncomment `INVSQRT_bitlen_scale` flag and comment other INVSQRT flags. */
// #define INVSQRT_10_9
#define INVSQRT_12_11
