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
#ifndef XFFT_H
#define XFFT_H

#include "xfft_work_size.h"
#include "xfft_plan.h"

void xfftCreateR2CPlan(xfftR2CPlan_t* plan, hipStream_t stream, int xx, int yy, int zz);
void xfftCreateC2RPlan(xfftC2RPlan_t* plan, hipStream_t stream, int xx, int yy, int zz);
void xfftReleaseR2CPlan(xfftR2CPlan_t* plan);
void xfftReleaseC2RPlan(xfftC2RPlan_t* plan);
void xfftR2CExecute(xfftR2CPlan_t* plan);
void xfftC2RExecute(xfftC2RPlan_t* plan);
bool xfftCheckProblemSizeAvailable(int xx, int yy, int zz);
#endif
