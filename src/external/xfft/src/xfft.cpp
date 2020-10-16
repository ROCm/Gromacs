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
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include "xfft.h"
#include "xfft_kernels.h"
#include "hip/hip_ext.h"

#define HIP_CHECK(status)                                                                          \
    if (status != hipSuccess) {                                                                    \
        std::cerr << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

typedef union
{
    float        f;
    unsigned int u;
    int          i;
}cb_t;

inline std::string xfftGetKernelPath(int xx, int yy, int zz)
{
    return std::to_string(xx) + "x" + std::to_string(yy) + "x" + std::to_string(zz) + "_";
}

int xfftGetR2CProblemSizeIndex(xfftR2CKernelWorkSize_t* xfftR2CKernelWorkSize, int xx, int yy, int zz) {
    for (int i = 0; i < sizeof(xfftR2CKernelWorkSize->problemSizePool)/sizeof(xfftR2CKernelWorkSize->problemSizePool[0]); i++) {
        if (xfftR2CKernelWorkSize->problemSizePool[i].xx == xx && 
            xfftR2CKernelWorkSize->problemSizePool[i].yy == yy &&
            xfftR2CKernelWorkSize->problemSizePool[i].zz == zz)
            return i;
    }
    return -1;
}

int xfftGetC2RProblemSizeIndex(xfftC2RKernelWorkSize_t* xfftC2RKernelWorkSize, int xx, int yy, int zz) {
    for (int i = 0; i < sizeof(xfftC2RKernelWorkSize->problemSizePool); i++) {
        if (xfftC2RKernelWorkSize->problemSizePool[i].xx == xx && 
            xfftC2RKernelWorkSize->problemSizePool[i].yy == yy &&
            xfftC2RKernelWorkSize->problemSizePool[i].zz == zz)
            return i;
    }
    return -1;
}

void xfftSetR2CKernelWorkSize(xfftR2CPlan_t* plan, int xx, int yy, int zz) {
    xfftR2CKernelWorkSize_t xfftR2CKernelWorkSize;
    int index = xfftGetR2CProblemSizeIndex(&xfftR2CKernelWorkSize, xx, yy, zz);

    //TODO for loop to find pool
    plan->wkSize1.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize1.globalWorkSizeX;
    plan->wkSize1.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize1.globalWorkSizeY;
    plan->wkSize1.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize1.globalWorkSizeZ;
    plan->wkSize1.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize1.localWorkSizeX;
    plan->wkSize1.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize1.localWorkSizeY;
    plan->wkSize1.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize1.localWorkSizeZ;

    plan->wkSize2.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize2.globalWorkSizeX;
    plan->wkSize2.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize2.globalWorkSizeY;
    plan->wkSize2.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize2.globalWorkSizeZ;
    plan->wkSize2.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize2.localWorkSizeX;
    plan->wkSize2.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize2.localWorkSizeY;
    plan->wkSize2.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize2.localWorkSizeZ;

    plan->wkSize3.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize3.globalWorkSizeX;
    plan->wkSize3.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize3.globalWorkSizeY;
    plan->wkSize3.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize3.globalWorkSizeZ;
    plan->wkSize3.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize3.localWorkSizeX;
    plan->wkSize3.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize3.localWorkSizeY;
    plan->wkSize3.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize3.localWorkSizeZ;

    plan->wkSize4.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize4.globalWorkSizeX;
    plan->wkSize4.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize4.globalWorkSizeY;
    plan->wkSize4.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize4.globalWorkSizeZ;
    plan->wkSize4.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize4.localWorkSizeX;
    plan->wkSize4.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize4.localWorkSizeY;
    plan->wkSize4.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize4.localWorkSizeZ;

    plan->wkSize5.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize5.globalWorkSizeX;
    plan->wkSize5.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize5.globalWorkSizeY;
    plan->wkSize5.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize5.globalWorkSizeZ;
    plan->wkSize5.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize5.localWorkSizeX;
    plan->wkSize5.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize5.localWorkSizeY;
    plan->wkSize5.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize5.localWorkSizeZ;

    plan->wkSize6.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize6.globalWorkSizeX;
    plan->wkSize6.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize6.globalWorkSizeY;
    plan->wkSize6.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize6.globalWorkSizeZ;
    plan->wkSize6.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize6.localWorkSizeX;
    plan->wkSize6.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize6.localWorkSizeY;
    plan->wkSize6.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize6.localWorkSizeZ;

    plan->wkSize7.globalWorkSizeX = xfftR2CKernelWorkSize.problemSizePool[index].wkSize7.globalWorkSizeX;
    plan->wkSize7.globalWorkSizeY = xfftR2CKernelWorkSize.problemSizePool[index].wkSize7.globalWorkSizeY;
    plan->wkSize7.globalWorkSizeZ = xfftR2CKernelWorkSize.problemSizePool[index].wkSize7.globalWorkSizeZ;
    plan->wkSize7.localWorkSizeX  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize7.localWorkSizeX;
    plan->wkSize7.localWorkSizeY  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize7.localWorkSizeY;
    plan->wkSize7.localWorkSizeZ  = xfftR2CKernelWorkSize.problemSizePool[index].wkSize7.localWorkSizeZ;
}

void xfftCreateR2CPlan(xfftR2CPlan_t* plan, hipStream_t stream, int xx, int yy, int zz) {
    std::string path = xfftGetKernelPath(xx, yy, zz);

    //std::cout << "Initialize xFFT R2C plan (" << xx << "x" << yy << "x" << zz << ")" << std::endl;

    int SIZE_OF_INT_BUF, SIZE_OF_CONST_BUF;
    SIZE_OF_INT_BUF = xx*yy*(zz/2+1)*8; // sizeof(complex float)
    SIZE_OF_CONST_BUF = 32*4; //32 * sizeof(int)
    // LoadModule and GetFunction
    hipModule_t Module;

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"1-xfft.kernel.Stockham4.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function1), Module, "fft_fwd"));
    //std::cout << "func "<< plan->Function1 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"2-xfft.kernel.Transpose5.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function2), Module, "transpose_gcn"));
    //std::cout << "func "<< plan->Function2 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"3-xfft.kernel.Stockham6.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function3), Module, "fft_fwd"));
    //std::cout << "func "<< plan->Function3 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"4-xfft.kernel.Transpose7.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function4), Module, "transpose_gcn"));
    //std::cout << "func "<< plan->Function4 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"5-xfft.kernel.Transpose8.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function5), Module, "transpose_gcn"));
    //std::cout << "func "<< plan->Function5 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"6-xfft.kernel.Stockham9.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function6), Module, "fft_fwd"));
    //std::cout << "func "<< plan->Function6 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"7-xfft.kernel.Transpose10.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function7), Module, "transpose_gcn"));
    //std::cout << "func "<< plan->Function7 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipMalloc((void**)&(plan->d_IntBuf), SIZE_OF_INT_BUF));
    HIP_CHECK(hipMalloc((void**)&(plan->d_ConstBuf), SIZE_OF_CONST_BUF));

    cb_t constNum;
    constNum.u = 1;
    hipMemset(plan->d_ConstBuf, 0, SIZE_OF_CONST_BUF);
    hipMemcpyHtoD(plan->d_ConstBuf, &constNum, sizeof(constNum));

    plan->stream = stream;
    xfftSetR2CKernelWorkSize(plan, xx, yy, zz);
}

void xfftSetC2RKernelWorkSize(xfftC2RPlan_t* plan, int xx, int yy, int zz) {
    xfftC2RKernelWorkSize_t xfftC2RKernelWorkSize;
    int index = xfftGetC2RProblemSizeIndex(&xfftC2RKernelWorkSize, xx, yy, zz);

    plan->wkSize8.globalWorkSizeX = xfftC2RKernelWorkSize.problemSizePool[index].wkSize8.globalWorkSizeX;
    plan->wkSize8.globalWorkSizeY = xfftC2RKernelWorkSize.problemSizePool[index].wkSize8.globalWorkSizeY;
    plan->wkSize8.globalWorkSizeZ = xfftC2RKernelWorkSize.problemSizePool[index].wkSize8.globalWorkSizeZ;
    plan->wkSize8.localWorkSizeX  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize8.localWorkSizeX;
    plan->wkSize8.localWorkSizeY  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize8.localWorkSizeY;
    plan->wkSize8.localWorkSizeZ  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize8.localWorkSizeZ;

    plan->wkSize9.globalWorkSizeX = xfftC2RKernelWorkSize.problemSizePool[index].wkSize9.globalWorkSizeX;
    plan->wkSize9.globalWorkSizeY = xfftC2RKernelWorkSize.problemSizePool[index].wkSize9.globalWorkSizeY;
    plan->wkSize9.globalWorkSizeZ = xfftC2RKernelWorkSize.problemSizePool[index].wkSize9.globalWorkSizeZ;
    plan->wkSize9.localWorkSizeX  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize9.localWorkSizeX;
    plan->wkSize9.localWorkSizeY  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize9.localWorkSizeY;
    plan->wkSize9.localWorkSizeZ  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize9.localWorkSizeZ;

    plan->wkSize10.globalWorkSizeX = xfftC2RKernelWorkSize.problemSizePool[index].wkSize10.globalWorkSizeX;
    plan->wkSize10.globalWorkSizeY = xfftC2RKernelWorkSize.problemSizePool[index].wkSize10.globalWorkSizeY;
    plan->wkSize10.globalWorkSizeZ = xfftC2RKernelWorkSize.problemSizePool[index].wkSize10.globalWorkSizeZ;
    plan->wkSize10.localWorkSizeX  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize10.localWorkSizeX;
    plan->wkSize10.localWorkSizeY  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize10.localWorkSizeY;
    plan->wkSize10.localWorkSizeZ  = xfftC2RKernelWorkSize.problemSizePool[index].wkSize10.localWorkSizeZ;
}

void xfftCreateC2RPlan(xfftC2RPlan_t* plan, hipStream_t stream, int xx, int yy, int zz) {
    std::string path = xfftGetKernelPath(xx, yy, zz);
    //std::cout << "Initialize xFFT C2R plan (" << xx << "x" << yy << "x" << zz << ")" << std::endl;

    int SIZE_OF_INTC2R_BUF, SIZE_OF_CONST_BUF;
    SIZE_OF_INTC2R_BUF = xx*yy*(zz/2+1)*8; // sizeof(complex float)
    SIZE_OF_CONST_BUF = 32*4; //32 * sizeof(int)

    hipModule_t Module;

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"8-xfft.kernel.Stockham11.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function8), Module, "fft_back"));
    //std::cout << "func "<< plan->Function8 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"9-xfft.kernel.Stockham13.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function9), Module, "fft_back"));
    //std::cout << "func "<< plan->Function9 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipModuleLoadData(&Module, kernel_map[path+"10-xfft.kernel.Stockham14.co"].data()));
    HIP_CHECK(hipModuleGetFunction(&(plan->Function10), Module, "fft_back"));
    //std::cout << "func "<< plan->Function10 << " hipModuleGetFunction: PASSED!\n";

    HIP_CHECK(hipMalloc(&(plan->d_IntC2RBuf), SIZE_OF_INTC2R_BUF));
    HIP_CHECK(hipMalloc(&(plan->d_ConstBuf), SIZE_OF_CONST_BUF));

    cb_t constNum;
    constNum.u = 1;
    hipMemset(plan->d_ConstBuf, 0, SIZE_OF_CONST_BUF);
    hipMemcpyHtoD(plan->d_ConstBuf, &constNum, sizeof(constNum));

    plan->stream = stream;
    xfftSetC2RKernelWorkSize(plan, xx, yy, zz);
}

void xfftR2CExecute(xfftR2CPlan_t* plan) {

        // Kernel1
        {
            plan->args1._d_ConstBuf = plan->d_ConstBuf;
            plan->args1._d_InputBuf = plan->d_InputBuf;
            plan->args1._d_OutputBuf = plan->d_OutputBuf;
            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args1), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size1),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config1, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function1, plan->wkSize1.globalWorkSizeX, plan->wkSize1.globalWorkSizeY, plan->wkSize1.globalWorkSizeZ,
                                     plan->wkSize1.localWorkSizeX, plan->wkSize1.localWorkSizeY, plan->wkSize1.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config1), 0));
        }

        // Kernel2
        {
            plan->args2._d_OutputBuf = plan->d_OutputBuf;
            plan->args2._d_IntBuf = plan->d_IntBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args2), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size2),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config2, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function2, plan->wkSize2.globalWorkSizeX, plan->wkSize2.globalWorkSizeY, plan->wkSize2.globalWorkSizeZ,
                                     plan->wkSize2.localWorkSizeX, plan->wkSize2.localWorkSizeY, plan->wkSize2.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config2), 0));
        }

        // Kernel3
        {
            plan->args3._d_ConstBuf = plan->d_ConstBuf;
            plan->args3._d_IntBuf = plan->d_IntBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args3), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size3),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config3, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function3, plan->wkSize3.globalWorkSizeX, plan->wkSize3.globalWorkSizeY, plan->wkSize3.globalWorkSizeZ,
                                     plan->wkSize3.localWorkSizeX, plan->wkSize3.localWorkSizeY, plan->wkSize3.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config3), 0));
        }

        // Kernel4
        {
            plan->args4._d_IntBuf = plan->d_IntBuf;
            plan->args4._d_OutputBuf = plan->d_OutputBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args4), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size4),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config4, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function4, plan->wkSize4.globalWorkSizeX, plan->wkSize4.globalWorkSizeY, plan->wkSize4.globalWorkSizeZ,
                                     plan->wkSize4.localWorkSizeX, plan->wkSize4.localWorkSizeY, plan->wkSize4.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config4), 0));
        }

        // Kernel5
        {
            plan->args5._d_OutputBuf = plan->d_OutputBuf;
            plan->args5._d_IntBuf = plan->d_IntBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args5), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size5),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config5, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function5, plan->wkSize5.globalWorkSizeX, plan->wkSize5.globalWorkSizeY, plan->wkSize5.globalWorkSizeZ,
                                     plan->wkSize5.localWorkSizeX, plan->wkSize5.localWorkSizeY, plan->wkSize5.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config5), 0));
        }

        // Kernel6
        {
            plan->args6._d_ConstBuf = plan->d_ConstBuf;
            plan->args6._d_IntBuf = plan->d_IntBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args6), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size6),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config6, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function6, plan->wkSize6.globalWorkSizeX, plan->wkSize6.globalWorkSizeY, plan->wkSize6.globalWorkSizeZ,
                                     plan->wkSize6.localWorkSizeX, plan->wkSize6.localWorkSizeY, plan->wkSize6.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config6), 0));
        }

        // Kernel7
        {
            plan->args7._d_OutputBuf = plan->d_OutputBuf;
            plan->args7._d_IntBuf = plan->d_IntBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args7), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size7),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config7, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function7, plan->wkSize7.globalWorkSizeX, plan->wkSize7.globalWorkSizeY, plan->wkSize7.globalWorkSizeZ,
                                     plan->wkSize7.localWorkSizeX, plan->wkSize7.localWorkSizeY, plan->wkSize7.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config7), 0));
        }
}

void xfftC2RExecute(xfftC2RPlan_t* plan ) {
        // Kernel8
        {
            plan->args8._d_ConstBuf = plan->d_ConstBuf;
            plan->args8._d_InputBuf = plan->d_InputBuf;
            plan->args8._d_IntC2RBuf = plan->d_IntC2RBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args8), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size8),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config8, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function8, plan->wkSize8.globalWorkSizeX, plan->wkSize8.globalWorkSizeY, plan->wkSize8.globalWorkSizeZ,
                                     plan->wkSize8.localWorkSizeX, plan->wkSize8.localWorkSizeY, plan->wkSize8.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config8), 0));
        }

        // Kernel9
        {
            plan->args9._d_ConstBuf = plan->d_ConstBuf;
            plan->args9._d_IntC2RBuf = plan->d_IntC2RBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args9), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size9),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config9, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function9, plan->wkSize9.globalWorkSizeX, plan->wkSize9.globalWorkSizeY, plan->wkSize9.globalWorkSizeZ,
                                     plan->wkSize9.localWorkSizeX, plan->wkSize9.localWorkSizeY, plan->wkSize9.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config9), 0));
        }

        // Kernel10
        {
            plan->args10._d_ConstBuf = plan->d_ConstBuf;
            plan->args10._d_IntC2RBuf = plan->d_IntC2RBuf;
            plan->args10._d_OutputBuf = plan->d_OutputBuf;

            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(plan->args10), HIP_LAUNCH_PARAM_BUFFER_SIZE, &(plan->size10),
                              HIP_LAUNCH_PARAM_END};
            memcpy(plan->config10, config, sizeof(config));
            HIP_CHECK(hipExtModuleLaunchKernel(plan->Function10, plan->wkSize10.globalWorkSizeX, plan->wkSize10.globalWorkSizeY, plan->wkSize10.globalWorkSizeZ,
                                     plan->wkSize10.localWorkSizeX, plan->wkSize10.localWorkSizeY, plan->wkSize10.localWorkSizeZ, 0, plan->stream, 
                                     NULL, (void**)&(plan->config10), 0));
        }
}

void xfftReleaseR2CPlan(xfftR2CPlan_t *plan)
{
    HIP_CHECK(hipFree(plan->d_IntBuf));
    HIP_CHECK(hipFree(plan->d_ConstBuf));
}

void xfftReleaseC2RPlan(xfftC2RPlan_t *plan)
{
    HIP_CHECK(hipFree(plan->d_IntC2RBuf));
    HIP_CHECK(hipFree(plan->d_ConstBuf));
}

bool xfftCheckProblemSizeAvailable(int xx, int yy, int zz)
{
    int index = -1;
    xfftR2CKernelWorkSize_t xfftR2CKernelWorkSize;
    index = xfftGetR2CProblemSizeIndex(&xfftR2CKernelWorkSize, xx, yy, zz);
    if(index < 0)
        return false;
    
    xfftC2RKernelWorkSize_t xfftC2RKernelWorkSize;
    index = xfftGetC2RProblemSizeIndex(&xfftC2RKernelWorkSize, xx, yy, zz);

    if(index < 0)
        return false;

    std::string kernel = xfftGetKernelPath(xx, yy, zz) +  "1-xfft.kernel.Stockham4.co";
    if ( kernel_map.find(kernel) != kernel_map.end())
        return true;
    return false;
}
