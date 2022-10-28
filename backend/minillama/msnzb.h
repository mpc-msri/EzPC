#pragma once

#include "mic.h"
#include "keypack.h"

// struct MSNZBKeyPack {
//     MICKeyPack micKey;
//     GroupElement r;
// };

std::pair<MSNZBKeyPack, MSNZBKeyPack> keyGenMSNZB(int bin, int bout, GroupElement rin, GroupElement rout, int start = 0, int end = -1);

GroupElement evalMSNZB(int party, int bin, int bout, GroupElement x, const MSNZBKeyPack &key, int start = 0, int end = -1, GroupElement *zcache = nullptr);