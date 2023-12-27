#pragma once

#include <llama/keypack.h>
#include "dcf.h"

template <typename T> using pair = std::pair<T,T>;

pair<ReluTruncateKeyPack> keyGenReluTruncate(int bin, int bout, int s, GroupElement rin, GroupElement routTruncate, GroupElement routRelu, GroupElement rout);

GroupElement evalRT_lrs(int party, GroupElement x, const ReluTruncateKeyPack &key, GroupElement &cache);

GroupElement evalRT_drelu(int party, GroupElement x, const ReluTruncateKeyPack &key, const GroupElement &cached);

GroupElement evalRT_mult(int party, GroupElement x, GroupElement y, const ReluTruncateKeyPack &key);
