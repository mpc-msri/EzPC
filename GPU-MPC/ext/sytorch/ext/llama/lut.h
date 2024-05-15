#pragma once

#include <llama/keypack.h>

std::pair<LUTKeyPack, LUTKeyPack> keyGenLUT(int bin, int bout, GroupElement rin, GroupElement rout);
std::pair<LUTSSKeyPack, LUTSSKeyPack> keyGenLUTSS(int bin, int bout, GroupElement rin, GroupElement rout);
std::pair<GroupElement, GroupElement> evalLUTSS_1(int party, GroupElement x, const std::vector<GroupElement> &tab, const LUTSSKeyPack &kp);
GroupElement evalLUTSS_2(int party, GroupElement res, GroupElement corr, const LUTSSKeyPack &kp);

std::pair<LUTDPFETKeyPack, LUTDPFETKeyPack> keyGenLUTDPFET(int bin, int bout, GroupElement rin, GroupElement routRes, GroupElement routCorr);
std::pair<GroupElement, GroupElement> evalLUTDPFET_1(int party, GroupElement x, const std::vector<GroupElement> &tab, LUTDPFETKeyPack &kp);
GroupElement evalLUTDPFET_2(int party, GroupElement res, GroupElement corr, const LUTDPFETKeyPack &kp);
