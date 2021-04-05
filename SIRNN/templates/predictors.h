// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void sirnnFixed(MYINT *X, int64_t* res);
void sirnnFloat(float **X, float* res);
void sirnnFixedSwitch(int i, MYINT** X, int32_t* res);

extern const int switches;
