#pragma once
#include <llama/keypack.h>

std::pair<ReluExtendKeyPack, ReluExtendKeyPack> keyGenReluExtend(int bin, int bout, GroupElement rin, GroupElement routX, GroupElement routDrelu);
