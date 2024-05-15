#pragma once

#include <llama/keypack.h>

std::pair<PubCmpKeyPack, PubCmpKeyPack> keyGenPubCmp(int bin, GroupElement rin, GroupElement rout);
GroupElement evalPubCmp(int party, GroupElement x, GroupElement c, const PubCmpKeyPack &key);
