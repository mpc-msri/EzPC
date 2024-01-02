#pragma once

#include <utility>
#include "mic.h"
#include "select.h"

void fill_pq(GroupElement *p, GroupElement *q, int n);
template <typename T>
using pair = std::pair<T, T>;

pair<FixToFloatKeyPack> keyGenFixToFloat(int bin, int scale, GroupElement rin, GroupElement *p, GroupElement *q);
void evalFixToFloat_1(int party, int bin, int scale, GroupElement x, const FixToFloatKeyPack &key, GroupElement *p, GroupElement *q, GroupElement &m, GroupElement &e, GroupElement &z, GroupElement &s, GroupElement &pow, GroupElement &sm);

GroupElement adjust(GroupElement m, GroupElement e);
GroupElement pow_helper(int scale,GroupElement y);
pair<FloatToFixKeyPack> keyGenFloatToFix(int bin, int scale, GroupElement rout);
