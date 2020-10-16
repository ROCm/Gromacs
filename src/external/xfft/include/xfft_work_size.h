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
#ifndef XFFT_WORK_SIZE_H
#define XFFT_WORK_SIZE_H


struct func_wkSize_t{
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t localWorkSizeX;
    uint32_t localWorkSizeY;
    uint32_t localWorkSizeZ;
};

class func_wkSize{
public:
    void set(int gwkX, int gwkY, int gwkZ, int lwkX, int lwkY, int lwkZ) {
        globalWorkSizeX = gwkX;
        globalWorkSizeY = gwkY;
        globalWorkSizeZ = gwkZ;
        localWorkSizeX = lwkX;
        localWorkSizeY = lwkY;
        localWorkSizeZ = lwkZ;
    }
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t localWorkSizeX;
    uint32_t localWorkSizeY;
    uint32_t localWorkSizeZ;
};

class _xfftR2CKernelWorkSize{
public:
    _xfftR2CKernelWorkSize(int _xx, int _yy, int _zz,
      int _k1gwkX, int _k1gwkY, int _k1gwkZ, int _k1lwkX, int _k1lwkY, int _k1lwkZ,
      int _k2gwkX, int _k2gwkY, int _k2gwkZ, int _k2lwkX, int _k2lwkY, int _k2lwkZ,
      int _k3gwkX, int _k3gwkY, int _k3gwkZ, int _k3lwkX, int _k3lwkY, int _k3lwkZ,
      int _k4gwkX, int _k4gwkY, int _k4gwkZ, int _k4lwkX, int _k4lwkY, int _k4lwkZ,
      int _k5gwkX, int _k5gwkY, int _k5gwkZ, int _k5lwkX, int _k5lwkY, int _k5lwkZ,
      int _k6gwkX, int _k6gwkY, int _k6gwkZ, int _k6lwkX, int _k6lwkY, int _k6lwkZ,
      int _k7gwkX, int _k7gwkY, int _k7gwkZ, int _k7lwkX, int _k7lwkY, int _k7lwkZ) {
        xx = _xx;
        yy = _yy;
        zz = _zz;
        wkSize1.set(_k1gwkX, _k1gwkY, _k1gwkZ, _k1lwkX, _k1lwkY, _k1lwkZ);
        wkSize2.set(_k2gwkX, _k2gwkY, _k2gwkZ, _k2lwkX, _k2lwkY, _k2lwkZ);
        wkSize3.set(_k3gwkX, _k3gwkY, _k3gwkZ, _k3lwkX, _k3lwkY, _k3lwkZ);
        wkSize4.set(_k4gwkX, _k4gwkY, _k4gwkZ, _k4lwkX, _k4lwkY, _k4lwkZ);
        wkSize5.set(_k5gwkX, _k5gwkY, _k5gwkZ, _k5lwkX, _k5lwkY, _k5lwkZ);
        wkSize6.set(_k6gwkX, _k6gwkY, _k6gwkZ, _k6lwkX, _k6lwkY, _k6lwkZ);
        wkSize7.set(_k7gwkX, _k7gwkY, _k7gwkZ, _k7lwkX, _k7lwkY, _k7lwkZ);
    }
    func_wkSize wkSize1, wkSize2, wkSize3, wkSize4, wkSize5, wkSize6, wkSize7;
    int xx, yy, zz;
};

class xfftR2CKernelWorkSize_t{
public:
    _xfftR2CKernelWorkSize problemSizePool[5] = {_xfftR2CKernelWorkSize(100,   100, 100,
                                                                  25020,     1,   1,  60,  1, 1,
                                                                     32,  1600,   1,  16, 16, 1,
                                                                  25500,     1,   1,  60,  1, 1,
                                                                     32,  1600,   1,  16, 16, 1,
                                                                     32,  1280,   1,  16, 16, 1,
                                                                  25500,     1,   1,  60,  1, 1,
                                                                     32,  1280,   1,  16, 16, 1),
                                                 _xfftR2CKernelWorkSize(84,     84,   84,
                                                                      7080, 1, 1, 60, 1, 1,
                                                                      32, 1344, 1, 16, 16, 1,
                                                                      7260, 1, 1, 60, 1, 1,
                                                                      32, 1344, 1, 16, 16, 1,
                                                                      32, 912, 1, 16, 16, 1,
                                                                      7260, 1, 1, 60, 1, 1,
                                                                      32, 912, 1, 16, 16, 1),
                                                 _xfftR2CKernelWorkSize(72,  72, 72,
                                                                   15624, 1, 1, 126, 1, 1,
                                                                       32, 1152, 1, 16, 16, 1,
                                                                   16002, 1, 1, 126, 1, 1,
                                                                       32, 1152, 1, 16, 16, 1,
                                                                       32, 672, 1, 16, 16, 1,
                                                                   16002, 1, 1, 126, 1, 1,
                                                                       32, 672, 1, 16, 16, 1),
                                                _xfftR2CKernelWorkSize(60, 60, 60,
                                                                       3648, 1, 1, 64, 1, 1,
                                                                       16, 960, 1, 16, 16, 1,
                                                                       3776, 1, 1, 64, 1, 1,
                                                                       16, 960, 1, 16, 16, 1,
                                                                       16, 480, 1, 16, 16, 1,
                                                                       3776, 1, 1, 64, 1, 1,
                                                                       16, 480, 1, 16, 16, 1),
                                                _xfftR2CKernelWorkSize(64, 64, 64,
                                                                       32768, 1, 1, 64, 1, 1,
                                                                       16, 1024, 1, 16, 16, 1,
                                                                       33792, 1, 1, 64, 1, 1,
                                                                       16, 1024, 1, 16, 16, 1,
                                                                       16, 528, 1, 16, 16, 1,
                                                                       33792, 1, 1, 64, 1, 1,
                                                                       16, 528, 1, 16, 16, 1),

                                                };
};

class _xfftC2RKernelWorkSize{
public:
    _xfftC2RKernelWorkSize(int _xx, int _yy, int _zz,
      int _k1gwkX, int _k1gwkY, int _k1gwkZ, int _k1lwkX, int _k1lwkY, int _k1lwkZ,
      int _k2gwkX, int _k2gwkY, int _k2gwkZ, int _k2lwkX, int _k2lwkY, int _k2lwkZ,
      int _k3gwkX, int _k3gwkY, int _k3gwkZ, int _k3lwkX, int _k3lwkY, int _k3lwkZ) {
        xx = _xx;
        yy = _yy;
        zz = _zz;
        wkSize8.set(_k1gwkX, _k1gwkY, _k1gwkZ, _k1lwkX, _k1lwkY, _k1lwkZ);
        wkSize9.set(_k2gwkX, _k2gwkY, _k2gwkZ, _k2lwkX, _k2lwkY, _k2lwkZ);
        wkSize10.set(_k3gwkX, _k3gwkY, _k3gwkZ, _k3lwkX, _k3lwkY, _k3lwkZ);
    }
    func_wkSize wkSize8, wkSize9, wkSize10;
    int xx, yy, zz;
};

class xfftC2RKernelWorkSize_t{
public:
    _xfftC2RKernelWorkSize problemSizePool[5] = {_xfftC2RKernelWorkSize(100,  100, 100,
                                                                    25500,  1,   1,  60, 1, 1,
                                                                    25500,  1,   1,  60, 1, 1,
                                                                    25020,  1,   1,  60, 1, 1),
                                                 _xfftC2RKernelWorkSize(84,    84,  84,
                                                                      7260,   1,   1,  60, 1, 1,
                                                                      7260,   1,   1,  60, 1, 1,
                                                                      7080,  1,   1, 60, 1, 1),
                                                 _xfftC2RKernelWorkSize(72,  72, 72,
                                                                      16002, 1, 1, 126, 1, 1,
                                                                      16002, 1, 1, 126, 1, 1,
                                                                      15624, 1, 1, 126, 1, 1),
                                                 _xfftC2RKernelWorkSize(60, 60, 60,
                                                                        3776, 1, 1, 64, 1, 1,
                                                                        3776, 1, 1, 64, 1, 1,
                                                                        3648, 1, 1, 64, 1, 1),
                                                 _xfftC2RKernelWorkSize(64, 64, 64,
                                                                        33792, 1, 1, 64, 1, 1,
                                                                        33792, 1, 1, 64, 1, 1,
                                                                        32768, 1, 1, 64, 1, 1),
                                                };
};

#endif
