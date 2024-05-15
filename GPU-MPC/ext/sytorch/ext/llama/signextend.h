#pragma once
#include <llama/keypack.h>

std::pair<SignExtend2KeyPack, SignExtend2KeyPack> keyGenSignExtend2(int bin, int bout, GroupElement rin, GroupElement rout);

std::pair<SlothSignExtendKeyPack, SlothSignExtendKeyPack> keyGenSlothSignExtend(int bin, int bout, GroupElement rin, GroupElement w, GroupElement rout);
GroupElement evalSlothSignExtend(int party, GroupElement x, GroupElement w, const SlothSignExtendKeyPack &kp);
