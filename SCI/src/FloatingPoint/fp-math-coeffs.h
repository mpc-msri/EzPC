/*
Authors: Deevashwer Rathee
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

#ifndef FLOATING_POINT_MATH_COEFFS_H__
#define FLOATING_POINT_MATH_COEFFS_H__
#include <vector>

// Coefficients for splines used in fp-math functions (derived from .poly files)
// z || s || 00 || e || {0,1} || m

std::vector<uint64_t> tan_N = {
    0x8000000000, 0x77c910800, 0x78c912702, 0x7996d0406, 0x79c91a312,
    0x79fb6802f,  0x7a96dd575, 0x7ab00996a, 0x7ac9393c5, 0x7ae26cc57,
    0x7afba4af5,  0x7b8a70bbe, 0x7b9711ce6, 0x7ba3b5ce8, 0x7bb05cfbb,
    0x7bbd0795a,  0x7bc9b5dc6, 0x7bd668104, 0x7be31e71c, 0x7befd941e,
    0x7bfc98c1d,  0x7c84ae99b, 0x7c8b136c3, 0x7c917af9b, 0x7c97e563a,
    0x7c9e52cba,  0x7ca4c353a, 0x7cab371dd, 0x7cb1ae4c8, 0x7cb829028,
    0x7cbea762a,  0x7cc529905, 0x7ccbafaf0, 0x7cd239e2b, 0x7cd8c84f8,
    0x7cdf5b1a1,  0x7ce5f2676, 0x7cec8e5ca, 0x7cf32f1f9, 0x7cf9d4d65,
    0x7d803fd3b,  0x7d8397dcd, 0x7d86f29a4, 0x7d8a501fe, 0x7d8db081f,
    0x7d9113d4c,  0x7d947a2cf, 0x7d97e39f9, 0x7d9b5041b, 0x7d9ec028c,
    0x7da2336aa,  0x7da5aa1d4, 0x7da924571, 0x7daca22ea, 0x7db023bb0,
    0x7db3a9138,  0x7db7324fc, 0x7dbabf87d, 0x7dbe50d42, 0x7dc1e64d5,
    0x7dc5800cb,  0x7dc91e2bc, 0x7dccc0c47, 0x7dd067f13, 0x7dd413ccd,
    0x7dd7c4729,  0x7ddb79fe4, 0x7ddf348c1, 0x7de2f438c, 0x7de6b9218,
    0x7dea83641,  0x7dee531ec, 0x7df228707, 0x7df603787, 0x7df9e456d,
    0x7dfdcb2c2,  0x7e80dc0ce, 0x7e82d5a0b, 0x7e84d262d, 0x7e86d264e,
    0x7e88d5b8d,  0x7e8adc70f, 0x7e8ce69ff, 0x7e8ef4590, 0x7e9105af8,
    0x7e931ab75,  0x7e953384d, 0x7e97502cc, 0x7e9970c45, 0x7e9b95611,
    0x7e9dbe194,  0x7e9feb038, 0x7ea21c36e, 0x7ea451cb1, 0x7ea68bd84,
    0x7ea8ca774,  0x7eab0dc15, 0x7ead55d07, 0x7eafa2bf2, 0x7eb1f4a8a,
    0x7eb44ba8a,  0x7eb6a7dbc, 0x7eb9095f3, 0x7ebb7050e, 0x7ebddccf7,
    0x7ec04efa5,  0x7ec2c6f1e, 0x7ec544d71, 0x7ec7c8cbe, 0x7eca52f32,
    0x7ecce3708,  0x7ecf7a68b, 0x7ed218015, 0x7ed4bc612, 0x7ed767afd,
    0x7eda1a162,  0x7edcd3be1, 0x7edf94d2b, 0x7ee25d805, 0x7ee52df49,
    0x7ee8065e4,  0x7eeae6eda, 0x7eedcfd45, 0x7ef0c1456, 0x7ef3bb758,
    0x7ef6be9ac,  0x7ef9caed0, 0x7efce0a5a};

std::vector<uint64_t> trig_knots_bits = {0x1, 0x2,  0x3,  0x4,  0x5, 0x6, 0x7,
                                         0x8, 0x9,  0xa,  0xb,  0xc, 0xd, 0xe,
                                         0xf, 0x10, 0x11, 0x12, 0x13};

std::vector<std::vector<uint64_t>> tan_coeffs = {
    // theta_1 (tan)
    {0x80c90fd8f, 0x80c90fd8c, 0x80c90fd7a, 0x80c90fd71, 0x80c90fd90,
     0x80c90fd8d, 0x80c90fd77, 0x80c90fd71, 0x80c90fd91, 0x80c90fd8e,
     0x80c90fd7a, 0x80c90fd71, 0x80c90fd8e, 0x80c90fd8f, 0x80c90fd7f,
     0x80c90fd75, 0x80c90fd91, 0x80c90fd90, 0x80c90fd83, 0x80c90fd76},
    // theta_3 (tan)
    {0x85bf324af, 0x8598a8bf8, 0x85a7132fa, 0x8595dd606, 0x83f917d57,
     0x83ce65aef, 0x83ec812ce, 0x83d3be6ed, 0x82f34c572, 0x82e0d33bb,
     0x82ef8a3ad, 0x82e5e993c, 0x82bb2caef, 0x82b3f2815, 0x82b608568,
     0x82b44645e, 0x82aa32c8e, 0x82a8d8d22, 0x82a914fe0, 0x82a91ab98}};

std::vector<uint64_t> exp2_knots_bits = {
    0x20, 0x40, 0x50, 0x60, 0x68, 0x70, 0x78, 0x80, 0x84, 0x88, 0x8c,
    0x90, 0x94, 0x98, 0x9c, 0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac,
    0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe, 0xc0, 0xc1,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc,
    0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf};

std::vector<std::vector<uint64_t>> exp2_coeffs = {
    // theta_0 (2^x)
    {0x7f8000008, 0x7f80000ce, 0x7f8000309, 0x7f8000783, 0x7f8000eb8,
     0x7f80019c0, 0x7f800294d, 0x7f8003dff, 0x7f8005782, 0x7f8007a37,
     0x7f800a3a7, 0x7f800d67f, 0x7f8011024, 0x7f8015816, 0x7f801a65f,
     0x7f802097e, 0x7f802709a, 0x7f802efdf, 0x7f8037c41, 0x7f80417b7,
     0x7f804c52f, 0x7f805894f, 0x7f8065c17, 0x7f80748da, 0x7f8084bb8,
     0x7f8096811, 0x7f80a9e7c, 0x7f80bf007, 0x7f80d5cb3, 0x7f80ee4b3,
     0x7f81092ca, 0x7f8126033, 0x7f8144e40, 0x7f81662d1, 0x7f818a337,
     0x7f81b01c8, 0x7f81d8d4f, 0x7f82046ff, 0x7f823377a, 0x7f826418d,
     0x7f8298453, 0x7f82d0033, 0x7f830b51e, 0x7f8348bd1, 0x7f838c135,
     0x7f83d2015, 0x7f8418cf7, 0x7f8468ed3, 0x7f84baff1, 0x7f850fc33,
     0x7f856bfec, 0x7f85caa12, 0x7f862c585, 0x7f86988e9, 0x7f87034c9,
     0x7f877a48b, 0x7f87f08ba, 0x7f8870f4b, 0x7f88f4fdd, 0x7f897eb41,
     0x7f8a10553, 0x7f8aa7c64, 0x7f8b46c15, 0x7f8be86e6},
    // theta_1 (2^x)
    {0x7eb16fda0, 0x7eb167048, 0x7eb158a95, 0x7eb144556, 0x7eb12b045,
     0x7eb10b328, 0x7eb0e5421, 0x7eb0b984d, 0x7eb089e27, 0x7eb04f90c,
     0x7eb010c2c, 0x7eafca59a, 0x7eaf80e56, 0x7eaf2bc91, 0x7eaed5a58,
     0x7eae6f4bc, 0x7eae0b8b7, 0x7ead971b6, 0x7ead1dd08, 0x7eac9e6e2,
     0x7eac17232, 0x7eab853ea, 0x7eaaef738, 0x7eaa4e38f, 0x7ea9a5250,
     0x7ea8f29be, 0x7ea83713c, 0x7ea7728f7, 0x7ea6a5ad6, 0x7ea5d0d8b,
     0x7ea4eee66, 0x7ea40437e, 0x7ea310a3e, 0x7ea211da3, 0x7ea10610c,
     0x7e9ff43e9, 0x7e9ed61f3, 0x7e9dabdb2, 0x7e9c7286e, 0x7e9b36ce0,
     0x7e99ec4c6, 0x7e9893a56, 0x7e972d963, 0x7e95c3557, 0x7e943ef03,
     0x7e92b4970, 0x7e912ddac, 0x7e8f7cc21, 0x7e8dca7b8, 0x7e8c13035,
     0x7e8a3e182, 0x7e886678a, 0x7e8688a72, 0x7e84813b8, 0x7e828a702,
     0x7e8063dc0, 0x7dfc94fbb, 0x7df81a003, 0x7df392ce0, 0x7deeee28a,
     0x7dea19cc0, 0x7de5290ff, 0x7de00e992, 0x7ddaf2f7f},
    // theta_2 (2^x)
    {0x7cf91ee2d, 0x7cfbc448f, 0x7cfe7be2a, 0x7d80a40a1, 0x7d820110b,
     0x7d836bba4, 0x7d84dad02, 0x7d864a1a4, 0x7d87ac0fc, 0x7d8932bc4,
     0x7d8aae4fa, 0x7d8c335e0, 0x7d8da8ee0, 0x7d8f3b04c, 0x7d90b58de,
     0x7d925bdd1, 0x7d93dd6af, 0x7d9587325, 0x7d972a096, 0x7d98cb3b4,
     0x7d9a71076, 0x7d9c22e14, 0x7d9dcc752, 0x7d9f836db, 0x7da13cf81,
     0x7da2fd340, 0x7da4c2379, 0x7da68bb82, 0x7da858062, 0x7daa261d9,
     0x7dac00cf2, 0x7dadde285, 0x7dafbe6b2, 0x7db1a5e90, 0x7db397777,
     0x7db585d3c, 0x7db77c5d3, 0x7db97a56e, 0x7dbb84266, 0x7dbd8488b,
     0x7dbf8fe62, 0x7dc1a4976, 0x7dc3c0f96, 0x7dc5d7139, 0x7dc80718d,
     0x7dca33023, 0x7dcc4e086, 0x7dce97506, 0x7dd0d5c66, 0x7dd30f590,
     0x7dd563533, 0x7dd7aee4e, 0x7dd9f6fc5, 0x7ddc66455, 0x7ddeb654d,
     0x7de1333bc, 0x7de395a56, 0x7de615f10, 0x7de891fd1, 0x7deb1332d,
     0x7deda335a, 0x7df037541, 0x7df2d6535, 0x7df56b505}};

// input in [2^-24, 1/32)
std::vector<uint64_t> log_knots_bits_1 = {0x1, 0x2, 0x3, 0x4,  0x5,  0x6,
                                          0x7, 0x8, 0x9, 0xa,  0xb,  0xc,
                                          0xd, 0xe, 0xf, 0x10, 0x11, 0x12};

std::vector<std::vector<uint64_t>> log2_coeffs_1 = {
    // theta_0 (exponent = -1)
    {0x67b8aa3b8, 0x454a480000, 0x45ab68c000, 0x45aaed8000, 0x45aff631de,
     0x45bbf3913f, 0x45c9f17880, 0x45da0ff60a, 0x45e9b4dbe7, 0x45f8244835,
     0x45ff7d9ca9, 0x460ff77051, 0x461f803dc0, 0x462ea08164, 0x463e357d1d,
     0x464e51cb78, 0x465f0c019d, 0x46782cc641, 0x468d45504f},
    // theta_1 (exponent = -1)
    {0x8000000000, 0x7fb8aa44d, 0x7fb8ad842, 0x7fb8abc67, 0x7fb8ab52d,
     0x7fb8ab092, 0x7fb8aae6a, 0x7fb8aaea1, 0x7fb8aae15, 0x7fb8aac8b,
     0x7fb8aac24, 0x7fb8aac55, 0x7fb8aac21, 0x7fb8aabae, 0x7fb8aab6f,
     0x7fb8aab7d, 0x7fb8aabf2, 0x7fb8aace2, 0x7fb8ab472},
    // theta_2 (exponent = -1)
    {0x8000000000, 0x8000000000, 0x4879d90000, 0x4858f8f8af, 0x483c24e766,
     0x48284f6d25, 0x480c17f59b, 0x47f9abb50f, 0x47cb85b74d, 0x7daa0befe,
     0x7e88ee318, 0x7ea06fdc5, 0x7eacc5cc3, 0x7eb305d49, 0x7eb5f00fe,
     0x7eb74acb9, 0x7eb7ee42a, 0x7eb83c33f, 0x7eb8319c5},
    // theta_3 (exponent = -1)
    {0x8000000000, 0x8000000000, 0x9b9aaaaaa, 0x988a04e05, 0x95baa0071,
     0x9383e0dac, 0x90d93ecbc, 0x8ee28ec73, 0x8ccfe5ed4, 0x8ab5cb2f8,
     0x88aefbf57, 0x86b0b062c, 0x84afc9cf7, 0x82ace3038, 0x80be0f23d,
     0x7f8de8437, 0x7ea71a1a4, 0x7e89da037, 0x7e8671d2f},
    // theta_0
    {0x8000000000, 0x68b8aa3a8, 0x4548100000, 0x45aea98000, 0x45b9dca492,
     0x45be186e9e, 0x45c9fcd81a, 0x45db6b8e3c, 0x45eb30dca8, 0x45f86d1e79,
     0x4608985af9, 0x460fe2a096, 0x461f98b981, 0x462f29c692, 0x463e55898d,
     0x464ecca0a2, 0x465e6dbdb3, 0x466c69df55, 0x465d4e7349},
    // theta_1
    {0x8000000000, 0x8000000000, 0x7fb8aa3c9, 0x7fb8ac575, 0x7fb8aba10,
     0x7fb8ab329, 0x7fb8aaeaa, 0x7fb8aafe9, 0x7fb8aaf94, 0x7fb8aacd6,
     0x7fb8aad02, 0x7fb8aac61, 0x7fb8aac2e, 0x7fb8aabf3, 0x7fb8aab7d,
     0x7fb8aabbe, 0x7fb8aab83, 0x7fb8aaa35, 0x7fb8aa370},
    // theta_2
    {0x8000000000, 0x8000000000, 0x8000000000, 0x485cca0000, 0x484871d41d,
     0x482bc5ed09, 0x481942d373, 0x480b644e28, 0x47fdfce81b, 0x47f8fd00f2,
     0x47eed0287d, 0x47ed147417, 0x47ec4a2dc9, 0x47ebe7d4f1, 0x47ebb68e32,
     0x47eba14fe5, 0x47eb958f93, 0x47eb8edc42, 0x47eb890fb2},
    // theta_3
    {0x8000000000, 0x8000000000, 0x8000000000, 0x98c800000, 0x95fe2be2c,
     0x93a4587e7, 0x90e765bc2, 0x8ef66dd2c, 0x8cec64b15, 0x8abc03a30,
     0x88beade75, 0x86b529589, 0x84b0ee221, 0x82b1e374f, 0x80bed09d2,
     0x7f9001cea, 0x7ea1e1c2f, 0x7e80b257d, 0x7deafa938}};

std::vector<std::vector<uint64_t>> ln_coeffs_1 = {
    // theta_0 (exponent = -1)
    {0x678000006, 0x453b380000, 0x459aaf8000, 0x459ef90000, 0x45aacf500f,
     0x45b885a2c2, 0x45bfe843e1, 0x45cf8a597b, 0x45ddc59dc0, 0x45ee920a6a,
     0x45fc1e1e10, 0x460c459015, 0x461b0350fc, 0x462b14e7e8, 0x463afee990,
     0x464ab0d034, 0x465a268804, 0x466aff7f0c, 0x468953cf7f},
    // theta_1 (exponent = -1)
    {0x8000000000, 0x7f8000056, 0x7f8001888, 0x7f800109a, 0x7f8000be9,
     0x7f800094c, 0x7f800089c, 0x7f800086e, 0x7f800076d, 0x7f80007dc,
     0x7f8000687, 0x7f800068a, 0x7f80005d9, 0x7f80005db, 0x7f80005d2,
     0x7f80005ae, 0x7f8000566, 0x7f800060e, 0x7f8000ba2},
    // theta_2 (exponent = -1)
    {0x8000000000, 0x8000000000, 0x4869280000, 0x484bd33333, 0x4838446bfc,
     0x481c2c7183, 0x480a199da5, 0x47efbcce98, 0x47c9a65fa4, 0x7c9fd543f,
     0x7db7022f1, 0x7ddbe1a14, 0x7defe5ab3, 0x7df7f9cb5, 0x7dfc01ef4,
     0x7dfe0bd55, 0x7dff10fd9, 0x7dff715d0, 0x7dff59cdb},
    // theta_3 (exponent = -1)
    {0x8000000000, 0x8000000000, 0x9a9000000, 0x97b333333, 0x94fd350fd,
     0x92c221e79, 0x90b0fd0d9, 0x8eadc492b, 0x8c981c397, 0x8aa05ee3e,
     0x8884b7b11, 0x8681fe997, 0x83e928c11, 0x81ef5d3e4, 0x8087377b2,
     0x7ec54fcf7, 0x7de152efe, 0x7dbdb0ae2, 0x7dba42662},
    // theta_0
    {0x8000000000, 0x67ffffffd, 0x451bc00000, 0x459fe37fff, 0x45a824a492,
     0x45af11c37d, 0x45bd4949df, 0x45cafa9d51, 0x45db06f325, 0x45e9d65f57,
     0x45f9bead01, 0x460a8c4c5a, 0x461a28113b, 0x4629d356c5, 0x4639a99321,
     0x46494ab8e4, 0x46589b8ca3, 0x465fee58f3, 0x463e1b1263},
    // theta_1
    {0x8000000000, 0x8000000000, 0x7efffffef, 0x7f8001261, 0x7f800091e,
     0x7f800083c, 0x7f8000724, 0x7f80005eb, 0x7f80005f5, 0x7f8000553,
     0x7f8000546, 0x7f80005b2, 0x7f8000568, 0x7f800053b, 0x7f8000529,
     0x7f80004ee, 0x7f8000493, 0x7f8000419, 0x7effffdf9},
    // theta_2
    {0x8000000000, 0x8000000000, 0x8000000000, 0x484dfbffff, 0x482dbf7df7,
     0x481c9e93e9, 0x480be1cf01, 0x47fc3967a0, 0x47f827aa0a, 0x47ebb9376b,
     0x47e9d7f264, 0x47e8fe8475, 0x47e8772e04, 0x47e839aace, 0x47e81c8560,
     0x47e80d87e4, 0x47e8063df8, 0x47e80293e3, 0x47dffd2ae2},
    // theta_3
    {0x8000000000, 0x8000000000, 0x8000000000, 0x97da00000, 0x94c7df7df,
     0x92a9c2f94, 0x908da0f81, 0x8dec80240, 0x8befd88b0, 0x89d7bcdaa,
     0x87d5b5bc0, 0x85e6d8667, 0x83d7894e7, 0x81d885d94, 0x7ff6ef62f,
     0x7eb5505ee, 0x7dd5b262f, 0x7db0f1a7e, 0x7da23bb13}};

// input in [1/32, 1)
std::vector<uint64_t> log_knots_bits_2 = {
    0x8,  0x10, 0x18, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x32, 0x34, 0x36, 0x38,
    0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f};

std::vector<std::vector<uint64_t>> log2_coeffs_2 = {
    // theta_0 (exponent = -1)
    {0x46be6a41ea, 0x46e862f26b, 0x46f891aa59, 0x471a2d2e39, 0x472f410b0c,
     0x4749ae4d10, 0x475af22c95, 0x476aceece2, 0x477a096dc5, 0x47890074cd,
     0x478fcd4013, 0x479d04a38f, 0x47aa81816e, 0x47b92524b7, 0x47bb4084e6,
     0x47bdca96d5, 0x47c84fdb85, 0x47ca3d2d5b, 0x47cc9a5d4d, 0x7f8000002,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_1 (exponent = -1)
    {0x7fb8af6df,  0x7fb8baf51,  0x7fb8c5f05,  0x7fb905041,  0x7fb97cf94,
     0x7fba5bb06,  0x7fbbdccf7,  0x7fbe1e940,  0x7fc19027f,  0x7fc6d6e63,
     0x7fceee0d6,  0x7fda01040,  0x7fe9e7a07,  0x80833fc4c,  0x808a7f2e9,
     0x8092f51ce,  0x809c1b742,  0x80a83ba5b,  0x80b6ab65f,  0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_2 (exponent = -1)
    {0x7eb74dce7,  0x7eb5a4b30,  0x7eb4898db,  0x7eaf5a57e,  0x7ea7f762e,
     0x7e9cf19c8,  0x7e8d1d58a,  0x7df1797e1,  0x7dbad70d0,  0x7ce085d81,
     0x7998dc93c,  0x47cec92823, 0x47e8f819d3, 0x47f8dc8aca, 0x47faefd620,
     0x47fd48bb3f, 0x47ffbee339, 0x4809764702, 0x480b4c62e8, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_3 (exponent = -1)
    {0x7e8e70556,  0x7e9877d7c,  0x7e9cf26cb,  0x7eaf38bb7,  0x7ec2aa537,
     0x7ed9f9141,  0x7ef5c1ac1,  0x7f8a37d9c,  0x7f9c48dd5,  0x7fb241222,
     0x7fcd6d46a,  0x7fec2dcbd,  0x8088be406,  0x80a55589e,  0x80b1ff4bc,
     0x80bfe0aa6,  0x80cdff0b6,  0x80dfc3c12,  0x80f3b1b03,  0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_0
    {0x467fe98efd, 0x46bdceffc4, 0x6e91e9dc2, 0x6fedbf7ca, 0x719c3e00c,
     0x72a3ee83a,  0x72f859bfa,  0x73ca200f3, 0x749c7cf76, 0x74e50da68,
     0x75a487668,  0x75dd4baa8,  0x7693ab45c, 0x769721217, 0x769721217,
     0x7697593a8,  0x7697593a8,  0x76b766181, 0x76b766181, 0x76dc282ce,
     0x76f63c2b6,  0x778c2df8c,  0x779e036b4, 0x77bf79f9b, 0x77c98c2c2,
     0x77f64926f,  0x77f87162f,  0x78868cbd4, 0x789c867a3, 0x78a5fc39b,
     0x78c9a973a,  0x78c4f67ed,  0x78e54fc48, 0x78ee7b1c6, 0x78df50698},
    // theta_1
    {0x7fb8a9e92, 0x7fb8ab86d, 0x7fb89a718, 0x7fb884883, 0x7fb85dfb7,
     0x7fb827a56, 0x7fb7f8279, 0x7fb7acdab, 0x7fb74fd15, 0x7fb6e3daf,
     0x7fb65fffc, 0x7fb5d7a64, 0x7fb53677e, 0x7fb5183c8, 0x7fb5183c8,
     0x7fb4f9b91, 0x7fb4f9b91, 0x7fb470198, 0x7fb470198, 0x7fb3dcd22,
     0x7fb36adb6, 0x7fb2e3966, 0x7fb25a3e9, 0x7fb18951c, 0x7fb128424,
     0x7fb030a32, 0x7faff1386, 0x7faf6367a, 0x7fae7e90e, 0x7fadfbe50,
     0x7facc12bf, 0x7faca35cb, 0x7fab8cfe9, 0x7fab11a08, 0x7fab29e9d},
    // theta_2
    {0x47eb857369, 0x47eb85a6fc, 0x47eb619c06, 0x47eb429519, 0x47eb19b5d2,
     0x47eaebe88a, 0x47eac8bd24, 0x47ea98d542, 0x47ea6557aa, 0x47ea305bd3,
     0x47e9f6da8e, 0x47e9c0b685, 0x47e9869a67, 0x47e977f0d1, 0x47e977f0d1,
     0x47e966b89e, 0x47e966b89e, 0x47e938aba0, 0x47e938aba0, 0x47e90a9863,
     0x47e8e649bb, 0x47e8be62c6, 0x47e897864a, 0x47e86411b4, 0x47e847dcfe,
     0x47e8114e43, 0x47dff9a277, 0x47dfb6a149, 0x47df5a2bfa, 0x47df1f40b4,
     0x47deafe7a1, 0x47de95abfc, 0x47de372940, 0x47de0503f0, 0x47ddf9511e},
    // theta_3
    {0x7de2ec016, 0x7ddf87418, 0x7dc61296d, 0x7db771277, 0x7da903ff2,
     0x7d9c29fbc, 0x7d939220f, 0x7d8982a84, 0x7d80185f5, 0x7cef00ce9,
     0x7cde644b1, 0x7cd028404, 0x7cc244055, 0x7cbe29067, 0x7cbe29067,
     0x7cb942c85, 0x7cb942c85, 0x7caf6931f, 0x7caf6931f, 0x7ca630243,
     0x7c9f00cc7, 0x7c97a1f74, 0x7c90c822f, 0x7c88a150d, 0x7c83d070d,
     0x7bf816a83, 0x7bf1250b0, 0x7be781b20, 0x7bdb97956, 0x7bd3a131e,
     0x7bc6d7df8, 0x7bc2a8805, 0x7bb8564df, 0x7bb258ba7, 0x7bafcba09}};

std::vector<std::vector<uint64_t>> ln_coeffs_2 = {
    // theta_0 (exponent = -1)
    {0x46bb9a1e7b, 0x46d9fb9cf8, 0x46ecb26fa5, 0x470d3c3a35, 0x472b1e3b6b,
     0x473d91b8bf, 0x474e945937, 0x475ee02d1b, 0x476e5a9db5, 0x477ca18e49,
     0x478abc30da, 0x4798d9660e, 0x479e2d5276, 0x47ae0a5cb9, 0x47b8908349,
     0x47ba5a7955, 0x47bcb4b84b, 0x47bf669326, 0x47c96c6b5c, 0x47c96c6b5c,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_1 (exponent = -1)
    {0x7f8004158,  0x7f800a32a,  0x7f8014387,  0x7f803bdbb,  0x7f8097f85,
     0x7f812ef0d,  0x7f8225ae5,  0x7f83c286a,  0x7f8652e08,  0x7f89eb796,
     0x7f8f2ecec,  0x7f96be35f,  0x7fa16248d,  0x7fbae32ca,  0x7fc557550,
     0x7fd114099,  0x7fe01d4a1,  0x7ff0d08e8,  0x8082cf41a,  0x8082cf41a,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_2 (exponent = -1)
    {0x7dfdeca5c,  0x7dfc34208,  0x7dfa11f0a,  0x7df389d94,  0x7de8303a7,
     0x7dd95aab5,  0x7dc4ff914,  0x7da7b8d0d,  0x7cfe4c5bb,  0x7c991b1df,
     0x799ae876c,  0x47c9bca86c, 0x47dbf44dc8, 0x47edc67f25, 0x47f85bb796,
     0x47f9f59952, 0x47fbf5830f, 0x47fe1cebea, 0x48085d276d, 0x48085d276d,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_3 (exponent = -1)
    {0x7dc705d51,  0x7dd1075fa,  0x7dda7901c,  0x7df185ae1,  0x7e87b933f,
     0x7e97494c5,  0x7ea936bbb,  0x7ebf6002e,  0x7eda240dd,  0x7ef7cf905,
     0x7f8d9c8ca,  0x7fa2c0a77,  0x7fbbec881,  0x7feecd82b,  0x80803413a,
     0x8089806ff,  0x8094d5438,  0x80a0ac48d,  0x80aeaf72c,  0x80aeaf72c,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000,
     0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000, 0x8000000000},
    // theta_0
    {0x468b3fff17, 0x468fedfc35, 0x6dc46258e, 0x6fb3f68bf, 0x70d29d1ff,
     0x71ce0e35b,  0x72bc4d051,  0x739d7db71, 0x73f184849, 0x748a00d7f,
     0x74b543494,  0x74f79786d,  0x75a19dd82, 0x75c764d58, 0x75c764d58,
     0x75ff84975,  0x75ff84975,  0x7697f2dc0, 0x7697f2dc0, 0x76bdb0a8a,
     0x76dc9f010,  0x77862c7cb,  0x7799cd708, 0x779f62216, 0x769e3f43a,
     0x76a80c72d,  0x76b597c68,  0x76c70e77b, 0x76f68235b, 0x75eac25a2,
     0x76bb75395,  0x77910ad64,  0x77863f3d8, 0x76c215d06, 0x7787f81a2},
    // theta_1
    {0x7f8000020, 0x7efffe852, 0x7effea8d5, 0x7effc8745, 0x7eff98787,
     0x7eff568f5, 0x7efef9fba, 0x7efe84de0, 0x7efdfcffb, 0x7efdbef7a,
     0x7efd3e86d, 0x7efc91f43, 0x7efbdc60e, 0x7efb2f804, 0x7efb2f804,
     0x7efa4b452, 0x7efa4b452, 0x7ef98a5ba, 0x7ef98a5ba, 0x7ef87f70c,
     0x7ef7a75db, 0x7ef67ddff, 0x7ef58ba08, 0x7ef525244, 0x7ef7b4cd5,
     0x7ef7243cf, 0x7ef6816ef, 0x7ef5cdb1e, 0x7ef4a844e, 0x7ef60b066,
     0x7ef48f793, 0x7ef2b2003, 0x7ef27c825, 0x7ef2e5aeb, 0x7ef16b1e5},
    // theta_2
    {0x47dffa5878, 0x47dff192bf, 0x47dfc7cca8, 0x47df98566c, 0x47df6588db,
     0x47df2c6190, 0x47dee9bf0c, 0x47dea1e738, 0x47de5905c6, 0x47de365d7a,
     0x47ddf92152, 0x47ddb038ca, 0x47dd69960b, 0x47dd2a1df6, 0x47dd2a1df6,
     0x47dcdea9ab, 0x47dcdea9ab, 0x47dca10e07, 0x47dca10e07, 0x47dc54320b,
     0x47dc179c5a, 0x47dbcbf1b0, 0x47db8fb2f8, 0x47db716c11, 0x47dbdf7a7d,
     0x47dbb2af4e, 0x47db83e351, 0x47db539d84, 0x47db12de70, 0x47db36b23d,
     0x47daeb235c, 0x47da94f813, 0x47da7c5c9c, 0x47da78aaf1, 0x47da3612dc},
    // theta_3
    {0x7d9e20c81, 0x7d9814a7e, 0x7d8976f1a, 0x7cfcde8b2, 0x7ceaf7ca2,
     0x7cda875a0, 0x7cca9da75, 0x7cbbfc0fa, 0x7caf01d8c, 0x7ca8d4244,
     0x7c9f58da6, 0x7c95466d3, 0x7c8c4d250, 0x7c84bcb8c, 0x7c84bcb8c,
     0x7bf925a7d, 0x7bf925a7d, 0x7bec62d58, 0x7bec62d58, 0x7bdde2a13,
     0x7bd2da518, 0x7bc6398c9, 0x7bbc788bf, 0x7bb72149e, 0x7bc33a7eb,
     0x7bbb8e109, 0x7bb3fc63c, 0x7bac962ee, 0x7ba3c7fc3, 0x7ba588ebe,
     0x7b9c29785, 0x7b923e57d, 0x7b8e65a4e, 0x7b8c7e905, 0x7b853d05b}};

std::vector<uint64_t> log2_int_to_float = {
    0x242ff, 0x242fe, 0x242fd, 0x242fc, 0x242fb, 0x242fa, 0x242f9, 0x242f8,
    0x242f7, 0x242f6, 0x242f5, 0x242f4, 0x242f3, 0x242f2, 0x242f1, 0x242f0,
    0x242ef, 0x242ee, 0x242ed, 0x242ec, 0x242eb, 0x242ea, 0x242e9, 0x242e8,
    0x242e7, 0x242e6, 0x242e5, 0x242e4, 0x242e3, 0x242e2, 0x242e1, 0x242e0,
    0x242df, 0x242de, 0x242dd, 0x242dc, 0x242db, 0x242da, 0x242d9, 0x242d8,
    0x242d7, 0x242d6, 0x242d5, 0x242d4, 0x242d3, 0x242d2, 0x242d1, 0x242d0,
    0x242cf, 0x242ce, 0x242cd, 0x242cc, 0x242cb, 0x242ca, 0x242c9, 0x242c8,
    0x242c7, 0x242c6, 0x242c5, 0x242c4, 0x242c3, 0x242c2, 0x242c1, 0x242c0,
    0x2427e, 0x2427c, 0x2427a, 0x24278, 0x24276, 0x24274, 0x24272, 0x24270,
    0x2426e, 0x2426c, 0x2426a, 0x24268, 0x24266, 0x24264, 0x24262, 0x24260,
    0x2425e, 0x2425c, 0x2425a, 0x24258, 0x24256, 0x24254, 0x24252, 0x24250,
    0x2424e, 0x2424c, 0x2424a, 0x24248, 0x24246, 0x24244, 0x24242, 0x24240,
    0x241fc, 0x241f8, 0x241f4, 0x241f0, 0x241ec, 0x241e8, 0x241e4, 0x241e0,
    0x241dc, 0x241d8, 0x241d4, 0x241d0, 0x241cc, 0x241c8, 0x241c4, 0x241c0,
    0x24178, 0x24170, 0x24168, 0x24160, 0x24158, 0x24150, 0x24148, 0x24140,
    0x240f0, 0x240e0, 0x240d0, 0x240c0, 0x24060, 0x24040, 0x23fc0, 0x40000,
    0x3fc0,  0x4040,  0x4060,  0x40c0,  0x40d0,  0x40e0,  0x40f0,  0x4140,
    0x4148,  0x4150,  0x4158,  0x4160,  0x4168,  0x4170,  0x4178,  0x41c0,
    0x41c4,  0x41c8,  0x41cc,  0x41d0,  0x41d4,  0x41d8,  0x41dc,  0x41e0,
    0x41e4,  0x41e8,  0x41ec,  0x41f0,  0x41f4,  0x41f8,  0x41fc,  0x4240,
    0x4242,  0x4244,  0x4246,  0x4248,  0x424a,  0x424c,  0x424e,  0x4250,
    0x4252,  0x4254,  0x4256,  0x4258,  0x425a,  0x425c,  0x425e,  0x4260,
    0x4262,  0x4264,  0x4266,  0x4268,  0x426a,  0x426c,  0x426e,  0x4270,
    0x4272,  0x4274,  0x4276,  0x4278,  0x427a,  0x427c,  0x427e,  0x42c0,
    0x42c1,  0x42c2,  0x42c3,  0x42c4,  0x42c5,  0x42c6,  0x42c7,  0x42c8,
    0x42c9,  0x42ca,  0x42cb,  0x42cc,  0x42cd,  0x42ce,  0x42cf,  0x42d0,
    0x42d1,  0x42d2,  0x42d3,  0x42d4,  0x42d5,  0x42d6,  0x42d7,  0x42d8,
    0x42d9,  0x42da,  0x42db,  0x42dc,  0x42dd,  0x42de,  0x42df,  0x42e0,
    0x42e1,  0x42e2,  0x42e3,  0x42e4,  0x42e5,  0x42e6,  0x42e7,  0x42e8,
    0x42e9,  0x42ea,  0x42eb,  0x42ec,  0x42ed,  0x42ee,  0x42ef,  0x42f0,
    0x42f1,  0x42f2,  0x42f3,  0x42f4,  0x42f5,  0x42f6,  0x42f7,  0x42f8,
    0x42f9,  0x42fa,  0x42fb,  0x42fc,  0x42fd,  0x42fe,  0x42ff,  0x4340};

std::vector<uint64_t> ln_int_to_float = {
    0x485b00f33c, 0x485aeac4f9, 0x485ad496b7, 0x485abe6874, 0x485aa83a31,
    0x485a920bee, 0x485a7bddab, 0x485a65af68, 0x485a4f8125, 0x485a3952e2,
    0x485a23249f, 0x485a0cf65c, 0x4859f6c819, 0x4859e099d6, 0x4859ca6b93,
    0x4859b43d50, 0x48599e0f0d, 0x485987e0ca, 0x485971b287, 0x48595b8444,
    0x4859455601, 0x48592f27be, 0x485918f97b, 0x485902cb38, 0x4858ec9cf5,
    0x4858d66eb2, 0x4858c0406f, 0x4858aa122c, 0x485893e3e9, 0x48587db5a6,
    0x4858678763, 0x4858515920, 0x48583b2add, 0x485824fc9a, 0x48580ece57,
    0x484ff14027, 0x484fc4e3a1, 0x484f98871b, 0x484f6c2a95, 0x484f3fce0f,
    0x484f137189, 0x484ee71503, 0x484ebab87d, 0x484e8e5bf7, 0x484e61ff71,
    0x484e35a2eb, 0x484e094665, 0x484ddce9df, 0x484db08d59, 0x484d8430d3,
    0x484d57d44d, 0x484d2b77c7, 0x484cff1b41, 0x484cd2bebb, 0x484ca66235,
    0x484c7a05af, 0x484c4da929, 0x484c214ca3, 0x484bf4f01d, 0x484bc89397,
    0x484b9c3711, 0x484b6fda8b, 0x484b437e05, 0x484b17217f, 0x484aeac4f9,
    0x484abe6874, 0x484a920bee, 0x484a65af68, 0x484a3952e2, 0x484a0cf65c,
    0x4849e099d6, 0x4849b43d50, 0x484987e0ca, 0x48495b8444, 0x48492f27be,
    0x484902cb38, 0x4848d66eb2, 0x4848aa122c, 0x48487db5a6, 0x4848515920,
    0x484824fc9a, 0x483ff14027, 0x483f98871b, 0x483f3fce0f, 0x483ee71503,
    0x483e8e5bf7, 0x483e35a2eb, 0x483ddce9df, 0x483d8430d3, 0x483d2b77c7,
    0x483cd2bebb, 0x483c7a05af, 0x483c214ca3, 0x483bc89397, 0x483b6fda8b,
    0x483b17217f, 0x483abe6874, 0x483a65af68, 0x483a0cf65c, 0x4839b43d50,
    0x48395b8444, 0x483902cb38, 0x4838aa122c, 0x4838515920, 0x482ff14027,
    0x482f3fce0f, 0x482e8e5bf7, 0x482ddce9df, 0x482d2b77c7, 0x482c7a05af,
    0x482bc89397, 0x482b17217f, 0x482a65af68, 0x4829b43d50, 0x482902cb38,
    0x4828515920, 0x481f3fce0f, 0x481ddce9df, 0x481c7a05af, 0x481b17217f,
    0x4819b43d50, 0x4818515920, 0x480ddce9df, 0x480b17217f, 0x4808515920,
    0x47fb17217f, 0x47eb17217f, 0x8000000000, 0x7eb17217f,  0x7fb17217f,
    0x808515920,  0x80b17217f,  0x80ddce9df,  0x818515920,  0x819b43d50,
    0x81b17217f,  0x81c7a05af,  0x81ddce9df,  0x81f3fce0f,  0x828515920,
    0x82902cb38,  0x829b43d50,  0x82a65af68,  0x82b17217f,  0x82bc89397,
    0x82c7a05af,  0x82d2b77c7,  0x82ddce9df,  0x82e8e5bf7,  0x82f3fce0f,
    0x82ff14027,  0x838515920,  0x838aa122c,  0x83902cb38,  0x8395b8444,
    0x839b43d50,  0x83a0cf65c,  0x83a65af68,  0x83abe6874,  0x83b17217f,
    0x83b6fda8b,  0x83bc89397,  0x83c214ca3,  0x83c7a05af,  0x83cd2bebb,
    0x83d2b77c7,  0x83d8430d3,  0x83ddce9df,  0x83e35a2eb,  0x83e8e5bf7,
    0x83ee71503,  0x83f3fce0f,  0x83f98871b,  0x83ff14027,  0x84824fc9a,
    0x848515920,  0x8487db5a6,  0x848aa122c,  0x848d66eb2,  0x84902cb38,
    0x8492f27be,  0x8495b8444,  0x84987e0ca,  0x849b43d50,  0x849e099d6,
    0x84a0cf65c,  0x84a3952e2,  0x84a65af68,  0x84a920bee,  0x84abe6874,
    0x84aeac4f9,  0x84b17217f,  0x84b437e05,  0x84b6fda8b,  0x84b9c3711,
    0x84bc89397,  0x84bf4f01d,  0x84c214ca3,  0x84c4da929,  0x84c7a05af,
    0x84ca66235,  0x84cd2bebb,  0x84cff1b41,  0x84d2b77c7,  0x84d57d44d,
    0x84d8430d3,  0x84db08d59,  0x84ddce9df,  0x84e094665,  0x84e35a2eb,
    0x84e61ff71,  0x84e8e5bf7,  0x84ebab87d,  0x84ee71503,  0x84f137189,
    0x84f3fce0f,  0x84f6c2a95,  0x84f98871b,  0x84fc4e3a1,  0x84ff14027,
    0x8580ece57,  0x85824fc9a,  0x8583b2add,  0x858515920,  0x858678763,
    0x8587db5a6,  0x85893e3e9,  0x858aa122c,  0x858c0406f,  0x858d66eb2,
    0x858ec9cf5,  0x85902cb38,  0x85918f97b,  0x8592f27be,  0x859455601,
    0x8595b8444,  0x85971b287,  0x85987e0ca,  0x8599e0f0d,  0x859b43d50,
    0x859ca6b93,  0x859e099d6,  0x859f6c819,  0x85a0cf65c,  0x85a23249f,
    0x85a3952e2,  0x85a4f8125,  0x85a65af68,  0x85a7bddab,  0x85a920bee,
    0x85aa83a31,  0x85abe6874,  0x85ad496b7,  0x85aeac4f9,  0x85b00f33c,
    0x85b17217f};

// input in [2^-24, 1/32)
std::vector<uint64_t> sin_knots_bits_1 = {0x1, 0x2, 0x3, 0x4,
                                          0x5, 0x6, 0x7, 0x8};

std::vector<std::vector<uint64_t>> sin_coeffs_1 = {
    // theta_0 (x)
    {0x80c90fdcb, 0x80c90fdc9, 0x80c90fdc8, 0x80c90fdcb, 0x80c90fdcd,
     0x80c90fdca, 0x80c90fdc8, 0x80c90fdc8, 0x80c90fdc8},
    // theta_1 (x^3)
    {0x4868a64cdd, 0x484914a934, 0x482cbdf0fd, 0x481e8199b3, 0x481b70f5b4,
     0x481a95452f, 0x481a6507d9, 0x481a59a9c6, 0x481a56d141},
    // theta_2 (x^5)
    {0x9fd5c1706, 0x9bc6f0f5c, 0x97c1ea8b3, 0x93d5ea0d8, 0x8fe266fb5,
     0x8bc8f1f91, 0x87c36e043, 0x83d6c62ff, 0x81822bf6a}};

std::vector<std::vector<uint64_t>> cos_coeffs_1 = {
    // theta_0 (x)
    {0x80c90fdd2, 0x80c90fdcc, 0x80c90fdce, 0x80c90fdcc, 0x80c90fdcc,
     0x80c90fdcc, 0x80c90fdcb, 0x80c90fdcb, 0x80c90fdcd},
    // theta_1 (x^3)
    {0x486a2a90c7, 0x4849f20c6d, 0x482e343a01, 0x481ea4041f, 0x481b6cd07b,
     0x481a99292b, 0x481a662452, 0x481a59ffbf, 0x481a56f965},
    // theta_2 (x^5)
    {0x9fff3e016, 0x9be0d5ef1, 0x97e6f0994, 0x93dd4afb2, 0x8fe199edf,
     0x8bd77b7ba, 0x87cfef880, 0x83e759e8e, 0x818a2e0fd}};

// input in [1/32, 0.5]
std::vector<uint64_t> sin_knots_bits_2 = {
    0x20, 0x30, 0x40, 0x48, 0x50, 0x58, 0x60, 0x64, 0x66, 0x68, 0x69,
    0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74,
    0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f};

std::vector<std::vector<uint64_t>> sin_coeffs_2 = {
    // theta_0 (x)
    {0x80c90fdca, 0x80c90fdef, 0x80c90fe23, 0x80c90fd59, 0x80c90fb69,
     0x80c90f550, 0x80c90e686, 0x80c90c854, 0x80c90b44e, 0x80c90933c,
     0x80c90a91b, 0x80c909392, 0x80c90749c, 0x80c905f25, 0x80c904104,
     0x80c901b8b, 0x80c8fef5b, 0x80c8fc240, 0x80c8f9294, 0x80c8f6146,
     0x80c8f2003, 0x80c8edf2a, 0x80c8e8cde, 0x80c8e47da, 0x80c8de7fe,
     0x80c8d8b1a, 0x80c8d20ae, 0x80c8cab1d, 0x80c8c2d50, 0x80c8ba53a,
     0x80c8b0cc7, 0x80c8a670c, 0x80c89cbf9, 0x80c89175d},
    // theta_1 (x^3)
    {0x481a561e85, 0x481a56097e, 0x481a55fa4e, 0x481a55ae60, 0x481a554d55,
     0x481a548cc6, 0x481a534bb1, 0x481a5166f1, 0x481a50571a, 0x481a4ece64,
     0x481a4f9381, 0x481a4ea17a, 0x481a4d6492, 0x481a4c8a52, 0x481a4b6f09,
     0x481a4a2501, 0x481a48b65e, 0x481a475120, 0x481a45e848, 0x481a448216,
     0x481a42c186, 0x481a411598, 0x481a3f110b, 0x481a3d6d07, 0x481a3b4156,
     0x481a393a6e, 0x481a36ff3b, 0x481a349fc8, 0x481a322d91, 0x481a2fa00c,
     0x481a2cde8c, 0x481a29fa4c, 0x481a275a94, 0x481a2465bc},
    // theta_2 (x^5)
    {0x80a977765, 0x80a4842fa, 0x80a3093a4, 0x80a173d7d, 0x80a04e353,
     0x809ed5d60, 0x809d2299d, 0x809b3b876, 0x809a55fe1, 0x809933182,
     0x8099a0117, 0x8098f9124, 0x8098300fb, 0x8097a7a9d, 0x809702dc5,
     0x80964e652, 0x8095911aa, 0x8094e0fbb, 0x809436e66, 0x809394dce,
     0x8092d4957, 0x809224753, 0x80915a0e6, 0x8090ba945, 0x808ff1714,
     0x808f3c713, 0x808e7cf6d, 0x808db8e9b, 0x808cf62d4, 0x808c321bf,
     0x808b661c2, 0x808a97804, 0x8089e1b2f, 0x80891b84b}};

std::vector<std::vector<uint64_t>> cos_coeffs_2 = {
    // theta_0 (x)
    {0x80c90fdc8, 0x80c90fdf1, 0x80c90fe1b, 0x80c90fd5f, 0x80c90fb79,
     0x80c90f557, 0x80c90e63d, 0x80c90c811, 0x80c90b44e, 0x80c909426,
     0x80c90a3f0, 0x80c9091f8, 0x80c90789c, 0x80c905efb, 0x80c903a1f,
     0x80c9016bc, 0x80c8ff36c, 0x80c8fc252, 0x80c8f90d3, 0x80c8f53fb,
     0x80c8f21b4, 0x80c8edb6d, 0x80c8e930d, 0x80c8e44ed, 0x80c8de0fd,
     0x80c8d871b, 0x80c8d1f33, 0x80c8caf7d, 0x80c8c2c3a, 0x80c8ba675,
     0x80c8b0c2a, 0x80c8a748c, 0x80c89c885, 0x80c891374},
    // theta_1 (x^3)
    {0x481a561a1e, 0x481a560b12, 0x481a55f75c, 0x481a55afc0, 0x481a554f73,
     0x481a548d87, 0x481a53465f, 0x481a51634e, 0x481a50571a, 0x481a4ed84a,
     0x481a4f5fe4, 0x481a4e928e, 0x481a4d8926, 0x481a4c88f1, 0x481a4b3604,
     0x481a49fef1, 0x481a48d515, 0x481a47519c, 0x481a45dc1b, 0x481a442941,
     0x481a42cc73, 0x481a40fe79, 0x481a3f35cf, 0x481a3d5c64, 0x481a3b1b0d,
     0x481a392555, 0x481a36f789, 0x481a34b573, 0x481a322859, 0x481a2fa5c3,
     0x481a2cdbc0, 0x481a2a34c3, 0x481a274c25, 0x481a245593},
    // theta_2 (x^5)
    {0x80a905221, 0x80a49729d, 0x80a2f8a68, 0x80a178bc7, 0x80a0528c4,
     0x809ed704c, 0x809d1c7a7, 0x809b3868f, 0x809a55fe1, 0x809939c73,
     0x80997fda8, 0x8098f0611, 0x809844f56, 0x8097a6f12, 0x8096e566a,
     0x80963b931, 0x80959f9aa, 0x8094e12f3, 0x8094319d0, 0x80936fc5b,
     0x8092d8fdf, 0x80921b852, 0x809167b6a, 0x8090b4ab4, 0x808fe45d7,
     0x808f357b0, 0x808e7a770, 0x808dbf9e2, 0x808cf49bb, 0x808c33c42,
     0x808b654fa, 0x808aa7528, 0x8089ddedc, 0x80891758c}};

std::vector<uint64_t> exp_knots_bits_1 = {0x1, 0x2, 0x3, 0x4,  0x5, 0x6,
                                          0x7, 0x8, 0x9, 0xa,  0xb, 0xc,
                                          0xd, 0xe, 0xf, 0x10, 0x11};

std::vector<std::vector<uint64_t>> exp_coeffs_1 = {
    // theta_0 (e^x)
    {0x7f80000ed, 0x7f8000120, 0x7f80000da, 0x7f80000e2, 0x7f80000ae,
     0x7f8000095, 0x7f8000087, 0x7f8000080, 0x7f800007e, 0x7f800007a,
     0x7f800007a, 0x7f800007b, 0x7f8000079, 0x7f800007a, 0x7f800007a,
     0x7f8000078, 0x7f800007e, 0x7f8000098},
    // theta_1 (e^x)
    {0x484a08009f, 0x483ba000b9, 0x481fd0a694, 0x480e91dc31, 0x47ed46fed6,
     0x7cd003cbf, 0x7ea216816, 0x7ed3111ce, 0x7ee9e2e10, 0x7ef54b16d,
     0x7efaa2b51, 0x7efd4c994, 0x7efeabd23, 0x7eff541a6, 0x7effaa77d,
     0x7effd5a81, 0x7effe98d2, 0x7efff121d},
    // theta_2 (e^x)
    {0x9ad6000f8, 0x98f800120, 0x96b406f5d, 0x94bc13de1, 0x92975ab85,
     0x9085e53b5, 0x8df937671, 0x8bef9c9db, 0x89ebf0c75, 0x87e48c9db,
     0x85e5d28fa, 0x83ea851b0, 0x81f2c31c1, 0x8092ad369, 0x7ef219188,
     0x7e9c6c2bf, 0x7e87dde61, 0x7e8341a52},
    // theta_0 (e^-x)
    {0x7f8000077, 0x7f8000052, 0x7f800005a, 0x7f8000048, 0x7f800003f,
     0x7f800003b, 0x7f8000037, 0x7f8000035, 0x7f8000034, 0x7f8000034,
     0x7f8000031, 0x7f8000038, 0x7f8000034, 0x7f8000036, 0x7f8000031,
     0x7f8000032, 0x7f800002c, 0x7f800000e},
    // theta_1 (e^-x)
    {0x483ba000e7, 0x48289852c5, 0x481a58ed75, 0x480a8f735a, 0x47fd987e3c,
     0x47fa95fa98, 0x47f93773aa, 0x47f89717e6, 0x47f84a148a, 0x47f82515e2,
     0x47f811834a, 0x47f809f36d, 0x47f804a384, 0x47f802663b, 0x47f80115f3,
     0x47f8008e05, 0x47f8003b8f, 0x47effffc8a},
    // theta_2 (e^-x)
    {0x99f800120, 0x97ac06e49, 0x95bc13c7b, 0x939140f5f, 0x90f0f9cac,
     0x8edc8e9f0, 0x8ccfaad40, 0x8ac9987e6, 0x88c5aeb9b, 0x86c635b61,
     0x84bccb91c, 0x82dc3038d, 0x80e61b0d3, 0x7fa665d7b, 0x7eae242e1,
     0x7e8bae8f2, 0x7e8218113, 0x7dfe60c96}};

std::vector<uint64_t> exp_knots_bits_2 = {
    0x20, 0x30, 0x40, 0x48, 0x50, 0x58, 0x60, 0x64, 0x68, 0x6c, 0x70,
    0x74, 0x78, 0x7c, 0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e,
    0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e, 0xa0, 0xa1, 0xa2,
    0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac};

std::vector<std::vector<uint64_t>> exp_coeffs_2 = {
    // theta_0 (e^x)
    {0x7f800016c, 0x7f8000648, 0x7f80010f1, 0x7f800234a, 0x7f800405d,
     0x7f80069eb, 0x7f800a32d, 0x7f800eebf, 0x7f8014f6e, 0x7f801c6c6,
     0x7f8025ab2, 0x7f8030b26, 0x7f803dedc, 0x7f804d717, 0x7f805f845,
     0x7f8074622, 0x7f808c41b, 0x7f80a75c3, 0x7f80c6123, 0x7f80e80cf,
     0x7f810dfeb, 0x7f813a654, 0x7f8169bfb, 0x7f819ebb3, 0x7f81d9a30,
     0x7f8218de0, 0x7f825f429, 0x7f82aaa3a, 0x7f82fd5d8, 0x7f8357854,
     0x7f83bae96, 0x7f8425040, 0x7f84956ad, 0x7f8512237, 0x7f8595048,
     0x7f8624042, 0x7f86beff5, 0x7f8760de3, 0x7f8811cae, 0x7f88cc39a,
     0x7f8992b1b, 0x7f8a65490, 0x7f8b454ee, 0x7f8cc753f},
    // theta_1 (e^x)
    {0x7effe9e54, 0x7effc601f, 0x7eff9030d, 0x7eff49d13, 0x7efeefc3e,
     0x7efe83efc, 0x7efe04023, 0x7efd6fd1f, 0x7efcc6ed5, 0x7efc0adf8,
     0x7efb36872, 0x7efa4e057, 0x7ef94c317, 0x7ef8332c3, 0x7ef7015d7,
     0x7ef5b62c8, 0x7ef451500, 0x7ef2d270a, 0x7ef1372fb, 0x7eef869f8,
     0x7eedba0f5, 0x7eebb7916, 0x7ee9aae3f, 0x7ee7780f1, 0x7ee51f6ba,
     0x7ee2b369d, 0x7ee01a6a3, 0x7edd6b9fb, 0x7eda937cd, 0x7ed794337,
     0x7ed46198e, 0x7ed1131e5, 0x7ecdad4de, 0x7eca04432, 0x7ec648e9e,
     0x7ec251cf8, 0x7ebe237ec, 0x7eb9e2f99, 0x7eb55b8a8, 0x7eb0b4551,
     0x7eabdeead, 0x7ea6dccd9, 0x7ea1a83ef, 0x7e98e56a6},
    // theta_2 (e^x)
    {0x7e8369342, 0x7e8577967, 0x7e87953f1, 0x7e89b016a, 0x7e8bdd562,
     0x7e8e0c7d6, 0x7e9047d62, 0x7e928ccd4, 0x7e94dab0c, 0x7e972b348,
     0x7e998cbfe, 0x7e9bf166e, 0x7e9e65514, 0x7ea0e18e8, 0x7ea3684b5,
     0x7ea5f954d, 0x7ea89419b, 0x7eab38253, 0x7eade884c, 0x7eb098d31,
     0x7eb353980, 0x7eb63ccd4, 0x7eb91379c, 0x7ebbfed57, 0x7ebefc62e,
     0x7ec1f44fb, 0x7ec50595c, 0x7ec813c26, 0x7ecb34deb, 0x7ece65236,
     0x7ed1afe9a, 0x7ed4fc0ab, 0x7ed8456a6, 0x7edbb56db, 0x7edf1d1f8,
     0x7ee2a1dd4, 0x7ee63dd4a, 0x7ee9d0d18, 0x7eed8694e, 0x7ef13dfc2,
     0x7ef5025e1, 0x7ef8d2167, 0x7efcb0e6c, 0x7f8186e6a},
    // theta_0 (e^-x)
    {0x7effffe33, 0x7effff6bf, 0x7efffe5ff, 0x7efffc8f4, 0x7efff9bb0,
     0x7efff5b5c, 0x7efff05de, 0x7effe964d, 0x7effe0d15, 0x7effd645a,
     0x7effc9b60, 0x7effbaf2c, 0x7effa9c49, 0x7eff963c9, 0x7eff8091e,
     0x7eff67d14, 0x7eff4c8ef, 0x7eff2e046, 0x7eff0d3ae, 0x7efee87de,
     0x7efec1dad, 0x7efe97234, 0x7efe6997d, 0x7efe38ab5, 0x7efe04e37,
     0x7efdccaf1, 0x7efd919f9, 0x7efd5305b, 0x7efd112d0, 0x7efccb4c4,
     0x7efc82591, 0x7efc353ec, 0x7efbe4fb2, 0x7efb9008a, 0x7efb3a250,
     0x7efadd784, 0x7efa7e217, 0x7efa1be29, 0x7ef9b4aa2, 0x7ef94ae86,
     0x7ef8dd754, 0x7ef86bb35, 0x7ef7f7038, 0x7ef852386},
    // theta_1 (e^-x)
    {0x47effeff3a, 0x47effd184b, 0x47effa4411, 0x47eff69795, 0x47eff2047c,
     0x47efec9ba6, 0x47efe67291, 0x47efdf6b28, 0x47efd7bc53, 0x47efcf3c78,
     0x47efc609a2, 0x47efbc226a, 0x47efb1810f, 0x47efa64a05, 0x47ef9aac15,
     0x47ef8e3b39, 0x47ef815670, 0x47ef73b2ff, 0x47ef65d3c7, 0x47ef5710e8,
     0x47ef484764, 0x47ef38ac62, 0x47ef28c235, 0x47ef18604f, 0x47ef07ba40,
     0x47eef65a64, 0x47eee4c66d, 0x47eed2cf17, 0x47eec08feb, 0x47eeadd7ff,
     0x47ee9aee82, 0x47ee87922a, 0x47ee7406bd, 0x47ee5ff42e, 0x47ee4c3b95,
     0x47ee378bf4, 0x47ee22d868, 0x47ee0e12b6, 0x47edf8cfb7, 0x47ede3928c,
     0x47edce2077, 0x47edb85e58, 0x47eda291ce, 0x47edb2ed12},
    // theta_2 (e^-x)
    {0x7dfa413cc, 0x7df65c7a2, 0x7df288c09, 0x7deed099b, 0x7deb1d72b,
     0x7de779a48, 0x7de3ec678, 0x7de061655, 0x7ddcf0640, 0x7dd9838b3,
     0x7dd6254d7, 0x7dd2d2f5b, 0x7dcf89074, 0x7dcc50bac, 0x7dc93379e,
     0x7dc6131b7, 0x7dc306781, 0x7dbffae81, 0x7dbd0bab3, 0x7dba14825,
     0x7db7401f1, 0x7db46665b, 0x7db19e95f, 0x7daee06c5, 0x7dac3342e,
     0x7da983cae, 0x7da6e625f, 0x7da4522ac, 0x7da1caf68, 0x7d9f4928e,
     0x7d9cd59a3, 0x7d9a67570, 0x7d98062cc, 0x7d95a7146, 0x7d93637ca,
     0x7d911479d, 0x7d8ed5215, 0x7d8ca2f9c, 0x7d8a72669, 0x7d8850712,
     0x7d8636960, 0x7d8421e65, 0x7d8218a41, 0x7d839007c}};

// input in [2^-11, 1)
std::vector<uint64_t> erf_knots_bits_1 = {
    0x8,  0x10, 0x14, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x21, 0x22, 0x23, 0x24,
    0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};

std::vector<std::vector<uint64_t>> erf_coeffs_1 = {
    // theta_0 (1)
    {0x45be3c7d08, 0x46985d4af5, 0x46cb25c4d0, 0x46ec194c1a, 0x470a9303a2,
     0x471c5d2d46, 0x472cbdbe58, 0x473ce69dc3, 0x474cf3fb0a, 0x475a63702d,
     0x476808c854, 0x476bcb2be9, 0x4778556d1b, 0x477b9beea0, 0x477ffc7988,
     0x478b24e800, 0x478de6a89f, 0x479b293cbd, 0x47a85c37e0, 0x47abd04ad1,
     0x47afef743e, 0x47ba4ce224, 0x47bcc0b5c0, 0x47bf3d7ea9},
    // theta_1 (x)
    {0x7f906ebaf, 0x7f906f56d, 0x7f90728d3, 0x7f907ad17, 0x7f908f5f3,
     0x7f90ae5c6, 0x7f90df5d9, 0x7f913487a, 0x7f91c8046, 0x7f926390c,
     0x7f93311e2, 0x7f9429fc8, 0x7f95545a0, 0x7f96c99cd, 0x7f989a736,
     0x7f9b0ca4b, 0x7f9d04aab, 0x7fa29ed4c, 0x7fa946b6e, 0x7fb0cb8f2,
     0x7fb905775, 0x7fc19bc8d, 0x7fc9fb812, 0x7fd1e6374},
    // theta_2 (x^2)
    {0x46ecbba7a5, 0x473f7e6e17, 0x475e97252b, 0x47787b3c90, 0x4788deefd1,
     0x478e7afee3, 0x479b09cda0, 0x47a867dbf9, 0x47acc01e59, 0x47b860f0a2,
     0x47bacab451, 0x47bd774bd1, 0x47c8377a94, 0x47c9f20cd4, 0x47cbf3cf93,
     0x47ce7b6fe4, 0x47d82d1e67, 0x47daa97eff, 0x47dd51bc64, 0x47e8063707,
     0x47e9648ee9, 0x47eab5d3c0, 0x47ebe6de12, 0x47ecf3ac60},
    // theta_3 (x^3)
    {0x47dc053d72, 0x47dbe646a2, 0x47dbb7c91b, 0x47db7c7233, 0x47db243c09,
     0x47dace6ddb, 0x47da6a7013, 0x47d9e52905, 0x47d936e8a6, 0x47d8aacbda,
     0x47d810ad54, 0x47cee8742a, 0x47cda6bfc9, 0x47cc495adb, 0x47cacf28a0,
     0x47c910ff28, 0x47bfc52546, 0x47b9e25e14, 0x47a872d734, 0x7887d00ab,
     0x7abd6b10c,  0x7ba3b767d,  0x7bdd99a9d,  0x7c869249a}};

// input in [1, 3.875]
std::vector<uint64_t> erf_knots_bits_2 = {
    0x2,  0x4,  0x6,  0x8,  0xa,  0xc,  0xe,  0x10, 0x12, 0x14, 0x16, 0x18,
    0x1a, 0x1c, 0x1e, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34,
    0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d};

std::vector<std::vector<uint64_t>> erf_coeffs_2 = {
    // theta_0 (1)
    {0x47c868ebc9, 0x47c94fbf20, 0x47c9e747bc, 0x47ca1d6071, 0x47c9e875db,
     0x47c927997f, 0x47bfdb43d3, 0x47bc1f6534, 0x47ae68a1b7, 0x47888b466a,
     0x7ab903a6c,  0x7bd536f02,  0x7cae967b1,  0x7cf7716f3,  0x7da22f4e0,
     0x7dc6a1dc8,  0x7de968995,  0x7e876154f,  0x7e98ab08f,  0x7ea8af4cd,
     0x7eb74001e,  0x7ec460bac,  0x7ed1661f7,  0x7edc755fa,  0x7ee6cfa2a,
     0x7eefaf0b3,  0x7ef7011a5,  0x7efd3de88,  0x7f804ef84,  0x7f81f11ea,
     0x7f8523621,  0x7f86557ab,  0x7f88001db,  0x7f893f962,  0x7f8ba2e08,
     0x7f8d88fe1,  0x7f8f55896,  0x7f8fdedea,  0x7f9587246,  0x7f97896c6,
     0x7f95fb467,  0x7f9de111f,  0x7facdffe3,  0x7f9aa4613,  0x7faf4dff6,
     0x7fb4b70e2},
    // theta_1 (x)
    {0x7fd6a5ec7,  0x7fdbcea5d,  0x7fdf08b82,  0x7fe02d869,  0x7fdf43175,
     0x7fdbe4cda,  0x7fd69c8cf,  0x7fcee13ad,  0x7fc51b417,  0x7fb964f94,
     0x7facd3214,  0x7f9f7b409,  0x7f90fb46a,  0x7f81fd706,  0x7ee56bdb5,
     0x7ec9729ed,  0x7eaf97aa4,  0x7e94acfb3,  0x7df91c518,  0x7dcdcd074,
     0x7da79a2e1,  0x7d8631e3e,  0x7ccbfc3aa,  0x7c9709100,  0x7bcde2d91,
     0x7afc6468e,  0x79fb2c971,  0x77aa9b3b4,  0x477f5d007f, 0x479a058703,
     0x47ab44ca75, 0x47ad46561f, 0x47b815cb54, 0x47b9161589, 0x47bb250126,
     0x47bcac863a, 0x47be10ad3c, 0x47be4595e2, 0x47c97d99d0, 0x47ca2f6bc7,
     0x47c9568d9e, 0x47cc788436, 0x47d9349db5, 0x47cabfefe6, 0x47d96210d0,
     0x47da4ae0de},
    // theta_2 (x^2)
    {0x47ed8bf3d0, 0x47ee296117, 0x47ee8729a6, 0x47eea8098e, 0x47ee928cd4,
     0x47ee424409, 0x47edc8f1f5, 0x47ed1e2bd6, 0x47ec4f028f, 0x47eb608fe4,
     0x47ea6a84f5, 0x47e96ec877, 0x47e866dc75, 0x47debf3c32, 0x47dcb94b29,
     0x47daef4a61, 0x47d9553a0c, 0x47cf6ee3e0, 0x47cca10836, 0x47ca308cb2,
     0x47c81a6643, 0x47bcaa4f54, 0x47b95931db, 0x47ad6b6606, 0x47a8c52a09,
     0x479a19265f, 0x4788caa625, 0x72f89708d,  0x778b3c3ca,  0x7882f8695,
     0x79828c60b,  0x7993a613c,  0x79adde30c,  0x79be2a578,  0x79e386f84,
     0x79fd0f0f4,  0x7a89931ea,  0x7a88df30e,  0x7ab27fa62,  0x7abc0ad33,
     0x7aa93fb89,  0x7ade12319,  0x7ba123bee,  0x7ab90ea5c,  0x7b9ed1252,
     0x7bab7b278},
    // theta_3 (x^3)
    {0x7c9349436,  0x7c9fcba89,  0x7ca6e424a,  0x7ca957ab5,  0x7ca80a57f,
     0x7ca30f324,  0x7c9bcdcf5,  0x7c91fb2da,  0x7c868bb97,  0x7bf3ce7b2,
     0x7bdab8f58,  0x7bc1fc77e,  0x7ba8f73bb,  0x7b90e3e93,  0x7af40f03a,
     0x7acd01212,  0x7aab2222a,  0x7a8a01504,  0x79dc61346,  0x79ad7d891,
     0x7986992a7,  0x78cd2a8b0,  0x7892e2def,  0x77cc14bd0,  0x76feead45,
     0x768889896,  0x74c53b50b,  0x4738912c1a, 0x474b0a91b2, 0x47589fed27,
     0x475fa2eebb, 0x476884ce05, 0x4769b759ac, 0x476a59ff67, 0x476c16b54a,
     0x476d27309b, 0x476e038813, 0x476dabd1e3, 0x4778bdec83, 0x47790a5129,
     0x476ff9e4cb, 0x477a4c3ec1, 0x477eb0d766, 0x47784bca01, 0x477e0019d9,
     0x477ee14486}};

const std::vector<uint64_t> sigmoid_bfloat16 = { 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16129, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16130, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16131, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16132, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16133, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16134, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16135, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16136, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16137, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16138, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16139, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16140, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16141, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16142, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16143, 16144, 16144, 16144, 16144, 16144, 16144, 16144, 16144, 16144, 16144, 16144, 16145, 16145, 16145, 16145, 16145, 16145, 16145, 16145, 16146, 16146, 16146, 16146, 16146, 16146, 16146, 16146, 16146, 16147, 16147, 16147, 16147, 16147, 16147, 16147, 16147, 16148, 16148, 16148, 16148, 16148, 16148, 16148, 16148, 16149, 16149, 16149, 16149, 16149, 16149, 16149, 16149, 16150, 16150, 16150, 16150, 16150, 16150, 16150, 16150, 16151, 16151, 16151, 16151, 16151, 16151, 16151, 16151, 16151, 16152, 16152, 16152, 16152, 16152, 16152, 16152, 16152, 16153, 16153, 16153, 16153, 16153, 16153, 16153, 16153, 16154, 16154, 16154, 16154, 16154, 16154, 16154, 16154, 16154, 16155, 16155, 16155, 16155, 16155, 16155, 16155, 16155, 16156, 16156, 16156, 16156, 16156, 16156, 16156, 16156, 16157, 16157, 16157, 16157, 16157, 16157, 16157, 16157, 16157, 16158, 16158, 16158, 16158, 16158, 16158, 16158, 16158, 16159, 16159, 16159, 16159, 16159, 16159, 16159, 16159, 16160, 16160, 16160, 16160, 16161, 16161, 16161, 16161, 16161, 16162, 16162, 16162, 16162, 16163, 16163, 16163, 16163, 16164, 16164, 16164, 16164, 16164, 16165, 16165, 16165, 16165, 16166, 16166, 16166, 16166, 16167, 16167, 16167, 16167, 16167, 16168, 16168, 16168, 16168, 16169, 16169, 16169, 16169, 16169, 16170, 16170, 16170, 16170, 16171, 16171, 16171, 16171, 16171, 16172, 16172, 16172, 16172, 16173, 16173, 16173, 16173, 16173, 16174, 16174, 16174, 16174, 16175, 16175, 16175, 16175, 16175, 16176, 16176, 16176, 16176, 16176, 16177, 16177, 16177, 16177, 16178, 16178, 16178, 16178, 16178, 16179, 16179, 16179, 16179, 16179, 16180, 16180, 16180, 16180, 16180, 16181, 16181, 16181, 16181, 16182, 16182, 16182, 16182, 16182, 16183, 16183, 16183, 16183, 16183, 16184, 16184, 16184, 16184, 16184, 16185, 16185, 16185, 16185, 16185, 16186, 16186, 16186, 16186, 16186, 16187, 16187, 16187, 16187, 16188, 16188, 16188, 16189, 16189, 16189, 16190, 16190, 16191, 16191, 16191, 16192, 16192, 16193, 16193, 16193, 16194, 16194, 16194, 16195, 16195, 16195, 16196, 16196, 16197, 16197, 16197, 16198, 16198, 16198, 16199, 16199, 16199, 16200, 16200, 16200, 16201, 16201, 16201, 16202, 16202, 16202, 16203, 16203, 16203, 16204, 16204, 16204, 16205, 16205, 16205, 16206, 16206, 16206, 16207, 16207, 16207, 16207, 16208, 16208, 16208, 16209, 16209, 16209, 16210, 16210, 16210, 16210, 16211, 16211, 16211, 16212, 16212, 16212, 16212, 16213, 16213, 16213, 16214, 16214, 16214, 16214, 16215, 16215, 16215, 16216, 16216, 16216, 16216, 16217, 16217, 16217, 16217, 16218, 16218, 16218, 16218, 16219, 16219, 16219, 16219, 16220, 16220, 16220, 16220, 16221, 16221, 16221, 16221, 16221, 16222, 16222, 16222, 16222, 16223, 16223, 16223, 16223, 16224, 16224, 16224, 16224, 16224, 16225, 16225, 16225, 16225, 16225, 16226, 16226, 16227, 16227, 16228, 16228, 16228, 16229, 16229, 16229, 16230, 16230, 16231, 16231, 16231, 16232, 16232, 16232, 16233, 16233, 16233, 16234, 16234, 16234, 16235, 16235, 16235, 16235, 16236, 16236, 16236, 16237, 16237, 16237, 16237, 16238, 16238, 16238, 16238, 16239, 16239, 16239, 16239, 16240, 16240, 16240, 16240, 16241, 16241, 16241, 16241, 16241, 16242, 16242, 16242, 16242, 16243, 16243, 16243, 16243, 16243, 16243, 16244, 16244, 16244, 16244, 16244, 16245, 16245, 16245, 16245, 16245, 16245, 16246, 16246, 16246, 16246, 16246, 16246, 16246, 16247, 16247, 16247, 16247, 16247, 16247, 16247, 16248, 16248, 16248, 16248, 16248, 16248, 16248, 16248, 16248, 16249, 16249, 16249, 16249, 16249, 16249, 16249, 16249, 16249, 16250, 16250, 16250, 16250, 16250, 16250, 16250, 16250, 16250, 16250, 16250, 16251, 16251, 16251, 16251, 16251, 16251, 16251, 16251, 16251, 16251, 16251, 16251, 16252, 16252, 16252, 16252, 16252, 16252, 16252, 16252, 16253, 16253, 16253, 16253, 16253, 16253, 16253, 16253, 16253, 16253, 16253, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16254, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16255, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16256, 16128, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16127, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16126, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16125, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16124, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16123, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16122, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16121, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16120, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16119, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16118, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16117, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16116, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16115, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16114, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16113, 16112, 16112, 16112, 16112, 16112, 16112, 16112, 16112, 16112, 16112, 16112, 16112, 16111, 16111, 16111, 16111, 16111, 16111, 16111, 16111, 16110, 16110, 16110, 16110, 16110, 16110, 16110, 16110, 16109, 16109, 16109, 16109, 16109, 16109, 16109, 16109, 16108, 16108, 16108, 16108, 16108, 16108, 16108, 16108, 16107, 16107, 16107, 16107, 16107, 16107, 16107, 16107, 16106, 16106, 16106, 16106, 16106, 16106, 16106, 16106, 16105, 16105, 16105, 16105, 16105, 16105, 16105, 16105, 16104, 16104, 16104, 16104, 16104, 16104, 16104, 16104, 16103, 16103, 16103, 16103, 16103, 16103, 16103, 16103, 16102, 16102, 16102, 16102, 16102, 16102, 16102, 16102, 16101, 16101, 16101, 16101, 16101, 16101, 16101, 16101, 16100, 16100, 16100, 16100, 16100, 16100, 16100, 16100, 16099, 16099, 16099, 16099, 16099, 16099, 16099, 16099, 16099, 16098, 16098, 16098, 16098, 16098, 16098, 16098, 16098, 16097, 16097, 16097, 16097, 16097, 16097, 16097, 16097, 16096, 16096, 16096, 16096, 16096, 16095, 16095, 16095, 16095, 16094, 16094, 16094, 16094, 16093, 16093, 16093, 16093, 16092, 16092, 16092, 16092, 16092, 16091, 16091, 16091, 16091, 16090, 16090, 16090, 16090, 16089, 16089, 16089, 16089, 16088, 16088, 16088, 16088, 16087, 16087, 16087, 16087, 16086, 16086, 16086, 16086, 16085, 16085, 16085, 16085, 16084, 16084, 16084, 16084, 16083, 16083, 16083, 16083, 16082, 16082, 16082, 16082, 16082, 16081, 16081, 16081, 16081, 16080, 16080, 16080, 16080, 16079, 16079, 16079, 16079, 16078, 16078, 16078, 16078, 16077, 16077, 16077, 16077, 16076, 16076, 16076, 16076, 16076, 16075, 16075, 16075, 16075, 16074, 16074, 16074, 16074, 16073, 16073, 16073, 16073, 16072, 16072, 16072, 16072, 16071, 16071, 16071, 16071, 16070, 16070, 16070, 16070, 16070, 16069, 16069, 16069, 16069, 16068, 16068, 16068, 16068, 16067, 16067, 16067, 16067, 16066, 16066, 16066, 16066, 16066, 16065, 16065, 16064, 16064, 16063, 16063, 16062, 16062, 16062, 16061, 16061, 16060, 16060, 16059, 16059, 16058, 16058, 16057, 16057, 16056, 16056, 16056, 16055, 16055, 16054, 16054, 16053, 16053, 16052, 16052, 16051, 16051, 16051, 16050, 16050, 16049, 16049, 16048, 16048, 16047, 16047, 16046, 16046, 16046, 16045, 16045, 16044, 16044, 16043, 16043, 16042, 16042, 16042, 16041, 16041, 16040, 16040, 16039, 16039, 16038, 16038, 16038, 16037, 16037, 16036, 16036, 16035, 16035, 16035, 16034, 16034, 16033, 16033, 16032, 16032, 16032, 16031, 16031, 16030, 16030, 16029, 16029, 16029, 16028, 16028, 16027, 16027, 16026, 16026, 16026, 16025, 16025, 16024, 16024, 16023, 16023, 16023, 16022, 16022, 16021, 16021, 16021, 16020, 16020, 16019, 16019, 16019, 16018, 16018, 16017, 16017, 16016, 16016, 16016, 16015, 16015, 16014, 16014, 16014, 16013, 16013, 16012, 16012, 16012, 16011, 16011, 16010, 16010, 16010, 16009, 16008, 16007, 16007, 16006, 16005, 16004, 16003, 16003, 16002, 16001, 16000, 15999, 15998, 15996, 15995, 15993, 15992, 15991, 15989, 15988, 15986, 15985, 15983, 15982, 15980, 15979, 15978, 15976, 15975, 15973, 15972, 15971, 15969, 15968, 15967, 15965, 15964, 15962, 15961, 15960, 15958, 15957, 15956, 15955, 15953, 15952, 15951, 15949, 15948, 15947, 15946, 15944, 15943, 15942, 15941, 15939, 15938, 15937, 15936, 15934, 15933, 15932, 15931, 15930, 15928, 15927, 15926, 15925, 15924, 15923, 15921, 15920, 15919, 15918, 15917, 15916, 15915, 15914, 15912, 15911, 15910, 15909, 15908, 15907, 15906, 15905, 15904, 15903, 15902, 15901, 15900, 15899, 15898, 15897, 15896, 15895, 15894, 15893, 15892, 15891, 15890, 15889, 15888, 15887, 15886, 15885, 15884, 15883, 15882, 15881, 15880, 15879, 15878, 15877, 15877, 15876, 15875, 15874, 15873, 15872, 15870, 15869, 15867, 15865, 15864, 15862, 15860, 15857, 15853, 15850, 15847, 15844, 15841, 15838, 15835, 15831, 15828, 15826, 15823, 15820, 15817, 15814, 15811, 15809, 15806, 15803, 15801, 15798, 15795, 15793, 15790, 15788, 15785, 15783, 15781, 15778, 15776, 15774, 15771, 15769, 15767, 15765, 15763, 15761, 15758, 15756, 15754, 15752, 15750, 15748, 15746, 15745, 15741, 15738, 15734, 15731, 15727, 15723, 15720, 15717, 15713, 15710, 15707, 15704, 15700, 15697, 15694, 15691, 15688, 15685, 15682, 15679, 15677, 15674, 15671, 15668, 15666, 15663, 15660, 15658, 15655, 15653, 15650, 15648, 15646, 15643, 15641, 15639, 15636, 15634, 15632, 15630, 15628, 15626, 15624, 15621, 15619, 15618, 15615, 15611, 15608, 15604, 15600, 15597, 15593, 15589, 15586, 15583, 15579, 15576, 15573, 15569, 15566, 15563, 15560, 15557, 15554, 15551, 15548, 15545, 15543, 15540, 15537, 15534, 15532, 15529, 15527, 15524, 15522, 15519, 15517, 15514, 15512, 15510, 15507, 15503, 15499, 15494, 15490, 15485, 15477, 15470, 15462, 15455, 15449, 15442, 15436, 15429, 15423, 15418, 15412, 15407, 15401, 15396, 15391, 15386, 15382, 15377, 15373, 15368, 15364, 15360, 15352, 15345, 15337, 15330, 15323, 15317, 15310, 15304, 15298, 15292, 15286, 15280, 15275, 15270, 15265, 15260, 15255, 15250, 15246, 15242, 15237, 15233, 15227, 15219, 15212, 15204, 15197, 15191, 15184, 15178, 15171, 15165, 15160, 15154, 15148, 15143, 15138, 15133, 15128, 15124, 15119, 15115, 15110, 15106, 15101, 15093, 15085, 15078, 15071, 15064, 15057, 15051, 15045, 15039, 15033, 15027, 15022, 15016, 15011, 15006, 15001, 14997, 14992, 14988, 14983, 14979, 14974, 14966, 14959, 14951, 14944, 14937, 14931, 14924, 14918, 14912, 14906, 14900, 14895, 14889, 14884, 14879, 14874, 14870, 14865, 14860, 14856, 14852, 14848, 14840, 14832, 14825, 14818, 14811, 14804, 14798, 14791, 14785, 14779, 14773, 14768, 14757, 14747, 14738, 14729, 14721, 14706, 14691, 14677, 14664, 14652, 14641, 14630, 14620, 14611, 14602, 14593, 14579, 14564, 14551, 14538, 14525, 14514, 14503, 14493, 14483, 14475, 14466, 14452, 14438, 14424, 14411, 14398, 14387, 14376, 14366, 14356, 14347, 14339, 14326, 14311, 14297, 14284, 14271, 14260, 14249, 14239, 14229, 14220, 14212, 14199, 14184, 14170, 14157, 14145, 14133, 14122, 14112, 14102, 14093, 14084, 14073, 14058, 14043, 14030, 14018, 14006, 13995, 13985, 13975, 13966, 13957, 13946, 13931, 13917, 13903, 13891, 13879, 13868, 13857, 13848, 13838, 13830, 13820, 13804, 13790, 13777, 13764, 13752, 13741, 13730, 13721, 13711, 13703, 13693, 13678, 13663, 13650, 13637, 13625, 13614, 13603, 13593, 13584, 13575, 13566, 13551, 13536, 13523, 13510, 13498, 13487, 13476, 13466, 13457, 13448, 13440, 13424, 13410, 13396, 13383, 13371, 13360, 13349, 13339, 13330, 13321, 13313, 13298, 13269, 13244, 13222, 13203, 13185, 13156, 13129, 13106, 13085, 13066, 13044, 13016, 12990, 12968, 12948, 12931, 12903, 12876, 12852, 12831, 12812, 12791, 12762, 12737, 12714, 12694, 12676, 12650, 12622, 12598, 12577, 12558, 12538, 12509, 12483, 12460, 12440, 12422, 12396, 12368, 12344, 12322, 12303, 12285, 12255, 12229, 12206, 12185, 12167, 12143, 12115, 12090, 12068, 12049, 12032, 12002, 11975, 11952, 11931, 11913, 11890, 11861, 11836, 11814, 11795, 11777, 11748, 11721, 11698, 11677, 11658, 11636, 11608, 11582, 11560, 11540, 11523, 11495, 11468, 11444, 11423, 11404, 11383, 11354, 11328, 11306, 11286, 11268, 11241, 11214, 11190, 11168, 11150, 11130, 11101, 11075, 11052, 11032, 11014, 10988, 10960, 10936, 10914, 10895, 10877, 10847, 10821, 10798, 10777, 10759, 10735, 10707, 10682, 10660, 10641, 10624, 10594, 10567, 10544, 10523, 10505, 10481, 10453, 10428, 10406, 10386, 10369, 10340, 10290, 10250, 10200, 10152, 10115, 10060, 10015, 9975, 9920, 9878, 9833, 9782, 9742, 9692, 9644, 9606, 9552, 9506, 9469, 9413, 9369, 9327, 9274, 9233, 9185, 9136, 9097, 9045, 8998, 8961, 8905, 8861, 8820, 8766, 8724, 8679, 8628, 8588, 8538, 8490, 8452, 8398, 8352, 8314, 8258, 8215, 8172, 8120, 8079, 8031, 7982, 7943, 7891, 7844, 7807, 7751, 7707, 7665, 7612, 7570, 7524, 7474, 7434, 7383, 7336, 7299, 7243, 7198, 7159, 7104, 7062, 7017, 6966, 6925, 6876, 6828, 6790, 6736, 6690, 6652, 6597, 6553, 6510, 6458, 6417, 6369, 6319, 6281, 6229, 6182, 6145, 6089, 6045, 6004, 5950, 5908, 5862, 5811, 5772, 5722, 5674, 5636, 5582, 5536, 5498, 5442, 5399, 5356, 5304, 5263, 5215, 5165, 5127, 5074, 5028, 4991, 4935, 4891, 4849, 4796, 4754, 4708, 4657, 4618, 4567, 4520, 4483, 4427, 4343, 4246, 4149, 4060, 3974, 3874, 3780, 3694, 3601, 3503, 3413, 3329, 3228, 3134, 3046, 2956, 2857, 2766, 2681, 2583, 2487, 2399, 2311, 2212, 2119, 2033, 1938, 1841, 1751, 1666, 1566, 1472, 1385, 1293, 1195, 1104, 1020, 921, 825, 737, 648, 550, 457, 372, 276, 179, 109, 66, 40, 24, 15, 9, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


#endif // FLOATING_POINT_MATH_COEFFS_H__
