#pragma once

#include <utility>
#include <llama/dcf.h>
#include <llama/keypack.h>

// struct BulkyLRSKeyPack
// {
//     DCFKeyPack dcfKeyN;
//     DCFKeyPack *dcfKeyS;
//     GroupElement *z;
//     GroupElement out;
// };

std::pair<BulkyLRSKeyPack, BulkyLRSKeyPack> keyGenBulkyLRS(int bin, int bout, int m, uint64_t *scales, GroupElement rin, GroupElement rout);

GroupElement evalBulkyLRS(int party, int bin, int bout, int m, uint64_t *scales, GroupElement x, const BulkyLRSKeyPack &key, int s, uint64_t scalar = 1);
