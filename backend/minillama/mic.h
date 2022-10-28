#pragma once
#include "dcf.h"
#include "keypack.h"

// struct MICKeyPack {
//     DCFKeyPack dcfKey;
//     GroupElement *z;
// };

std::pair<MICKeyPack, MICKeyPack> keyGenMIC(int bin, int bout, int m, uint64_t *p, uint64_t *q, GroupElement rin, GroupElement *rout);

void evalMIC(int party, int bin, int bout, int m, uint64_t *p, uint64_t *q, GroupElement x, const MICKeyPack &key, GroupElement *y);
