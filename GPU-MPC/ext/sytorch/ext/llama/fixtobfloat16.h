#pragma once

#include <llama/keypack.h>

std::pair<F2BF16KeyPack, F2BF16KeyPack> keyGenF2BF16(int bin, GroupElement rin, GroupElement rout);
std::pair<GroupElement, GroupElement> evalF2BF16_1(int party, GroupElement x, const F2BF16KeyPack &key);
GroupElement evalF2BF16_2(int party, GroupElement x, GroupElement k, GroupElement m, const F2BF16KeyPack &key);
GroupElement evalF2BF16_3(int party, GroupElement k, GroupElement xm, const F2BF16KeyPack &key);
