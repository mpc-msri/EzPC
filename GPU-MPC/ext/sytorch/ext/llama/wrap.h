#pragma once

#include <llama/keypack.h>

std::pair<WrapSSKeyPack, WrapSSKeyPack> keyGenWrapSS(int bin, GroupElement rin, GroupElement rout);
GroupElement evalWrapSS(int party, GroupElement x, const WrapSSKeyPack &key);
std::pair<WrapDPFKeyPack, WrapDPFKeyPack> keyGenWrapDPF(int bin, GroupElement rin, GroupElement rout);
GroupElement evalWrapDPF(int party, GroupElement x, const WrapDPFKeyPack &key);
