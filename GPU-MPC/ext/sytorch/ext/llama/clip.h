#pragma once

#include <llama/keypack.h>

std::pair<ClipKeyPack, ClipKeyPack> keyGenClip(int bin, GroupElement rin, GroupElement rout);
GroupElement evalClip_1(int party, int maxBw, GroupElement x, const ClipKeyPack &key);
GroupElement evalClip_2(int party, int maxBw, GroupElement x,  GroupElement y, const ClipKeyPack &key);
