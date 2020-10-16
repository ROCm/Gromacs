/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef XFFT_PLAN_H
#define XFFT_PLAN_H

#include "hip/hip_runtime.h"

struct args1_t{
    void* _d_ConstBuf;
    void* _d_InputBuf;
    void* _d_OutputBuf;
};

struct args2_t{
    void* _d_OutputBuf;
    void* _d_IntBuf;
};
struct args3_t{
    void* _d_ConstBuf;
    void* _d_IntBuf;
};
struct args4_t{
    void* _d_IntBuf;
    void* _d_OutputBuf;
};
struct args5_t{
    void* _d_OutputBuf;
    void* _d_IntBuf;
};
struct args6_t{
    void* _d_ConstBuf;
    void* _d_IntBuf;
};
struct args7_t{
    void* _d_IntBuf;
    void* _d_OutputBuf;
};

struct xfftR2CPlan_t{
    args1_t args1;
    args2_t args2;
    args3_t args3;
    args4_t args4;
    args5_t args5;
    args6_t args6;
    args7_t args7;
    size_t size1 = sizeof(args1);
    size_t size2 = sizeof(args2);
    size_t size3 = sizeof(args3);
    size_t size4 = sizeof(args4);
    size_t size5 = sizeof(args5);
    size_t size6 = sizeof(args6);
    size_t size7 = sizeof(args7);
    void *config1[5], *config2[5], *config3[5], *config4[5], *config5[5], *config6[5], *config7[5];
    hipDeviceptr_t d_InputBuf, d_OutputBuf, d_IntBuf, d_ConstBuf;
    hipFunction_t Function1, Function2, Function3, Function4, Function5, Function6, Function7;
    func_wkSize_t wkSize1, wkSize2, wkSize3, wkSize4, wkSize5, wkSize6, wkSize7;
    hipStream_t stream;
};


struct args8_t{
    void* _d_ConstBuf;
    void* _d_InputBuf;
    void* _d_IntC2RBuf;
};
struct args9_t{
    void* _d_ConstBuf;
    void* _d_IntC2RBuf;
};
struct args10_t{
    void* _d_ConstBuf;
    void* _d_IntC2RBuf;
    void* _d_OutputBuf;
};

struct xfftC2RPlan_t{
    args8_t args8;
    args9_t args9;
    args10_t args10;
    size_t size8 = sizeof(args8);
    size_t size9 = sizeof(args9);
    size_t size10 = sizeof(args10);
    void *config8[5], *config9[5], *config10[5];
    hipDeviceptr_t d_InputBuf, d_OutputBuf, d_IntC2RBuf, d_ConstBuf;
    hipFunction_t Function8, Function9, Function10;
    func_wkSize_t wkSize8, wkSize9, wkSize10;
    hipStream_t stream;
};

#endif

