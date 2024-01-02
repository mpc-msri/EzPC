#pragma once
#include <llama/keypack.h>

// 64 ANDs at a time
std::pair<BitwiseAndKeyPack, BitwiseAndKeyPack> keyGenBitwiseAnd(GroupElement rin1, GroupElement rin2, GroupElement rout);
GroupElement evalBitwiseAnd(int party, GroupElement x, GroupElement y, const BitwiseAndKeyPack &key);
GroupElement evalAnd(int party, GroupElement x, GroupElement y, const BitwiseAndKeyPack &key);
