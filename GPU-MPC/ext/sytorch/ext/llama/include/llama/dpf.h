#pragma once
#include <cryptoTools/Common/Defines.h>
#include <llama/group_element.h>
#include <llama/keypack.h>

std::pair<DPFKeyPack, DPFKeyPack> keyGenDPF(int bin, int bout, GroupElement idx, GroupElement payload);
GroupElement evalDPF_EQ(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_GT(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_LT(int party, DPFKeyPack &key, GroupElement x);
void evalAll(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out);
GroupElement evalAll_reduce(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);

std::pair<DPFETKeyPack, DPFETKeyPack> keyGenDPFET(int bin, GroupElement idx);
std::pair<GroupElement, GroupElement> evalAll_reduce_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);
GroupElement evalDPFET_LT(int party, const DPFETKeyPack &key, GroupElement x);
