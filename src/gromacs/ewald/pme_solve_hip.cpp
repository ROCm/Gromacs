#include "hip/hip_runtime.h"
/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018,2019,2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \internal \file
 *  \brief Implements PME GPU Fourier grid solving in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <cassert>

#include <cmath>

#include "gromacs/gpu_utils/hip_arch_utils_hip.h"

#include "pme_hip.h"
#include "gromacs/gpu_utils/cuda_kernel_utils_hip.h"

template<class T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
__device__ inline
T warp_move_dpp(const T& input) {
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    struct V { int words[words_no]; };
    V a = __builtin_bit_cast(V, input);

    #pragma unroll
    for (int i = 0; i < words_no; i++) {
        a.words[i] = __builtin_amdgcn_update_dpp(
          0, a.words[i],
          dpp_ctrl, row_mask, bank_mask, bound_ctrl
        );
    }

    return __builtin_bit_cast(T, a);
}

/*! \brief
 * PME complex grid solver kernel function.
 *
 * \tparam[in] gridOrdering             Specifies the dimension ordering of the complex grid.
 * \tparam[in] computeEnergyAndVirial   Tells if the reciprocal energy and virial should be computed.
 * \tparam[in] gridIndex                The index of the grid to use in the kernel.
 * \param[in]  kernelParams             Input PME CUDA data in constant memory.
 */
template<GridOrdering gridOrdering, bool computeEnergyAndVirial, const int gridIndex>
LAUNCH_BOUNDS_EXACT_SINGLE(c_solveMaxThreadsPerBlock) CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE __global__
        void pme_solve_kernel(const struct PmeGpuHipKernelParams kernelParams)
{
    /* This kernel supports 2 different grid dimension orderings: YZX and XYZ */
    int majorDim, middleDim, minorDim;
    switch (gridOrdering)
    {
        case GridOrdering::YZX:
            majorDim  = YY;
            middleDim = ZZ;
            minorDim  = XX;
            break;

        case GridOrdering::XYZ:
            majorDim  = XX;
            middleDim = YY;
            minorDim  = ZZ;
            break;

        default: assert(false);
    }

    /* Global memory pointers */
    const float* __restrict__ gm_splineValueMajor = kernelParams.grid.d_splineModuli[gridIndex]
                                                    + kernelParams.grid.splineValuesOffset[majorDim];
    const float* __restrict__ gm_splineValueMiddle = kernelParams.grid.d_splineModuli[gridIndex]
                                                     + kernelParams.grid.splineValuesOffset[middleDim];
    const float* __restrict__ gm_splineValueMinor = kernelParams.grid.d_splineModuli[gridIndex]
                                                    + kernelParams.grid.splineValuesOffset[minorDim];
    float* __restrict__ gm_virialAndEnergy = kernelParams.constants.d_virialAndEnergy[gridIndex];
    float2* __restrict__ gm_grid           = (float2*)kernelParams.grid.d_fourierGrid[gridIndex];

    /* Various grid sizes and indices */
    const int localOffsetMinor = 0, localOffsetMajor = 0, localOffsetMiddle = 0; // unused
    const int localSizeMinor   = kernelParams.grid.complexGridSizePadded[minorDim];
    const int localSizeMiddle  = kernelParams.grid.complexGridSizePadded[middleDim];
    const int localCountMiddle = kernelParams.grid.complexGridSize[middleDim];
    const int localCountMinor  = kernelParams.grid.complexGridSize[minorDim];
    const int nMajor           = kernelParams.grid.realGridSize[majorDim];
    const int nMiddle          = kernelParams.grid.realGridSize[middleDim];
    const int nMinor           = kernelParams.grid.realGridSize[minorDim];
    const int maxkMajor        = (nMajor + 1) / 2;  // X or Y
    const int maxkMiddle       = (nMiddle + 1) / 2; // Y OR Z => only check for !YZX
    const int maxkMinor        = (nMinor + 1) / 2;  // Z or X => only check for YZX

    /* Each thread works on one cell of the Fourier space complex 3D grid (gm_grid).
     * Each block handles up to c_solveMaxThreadsPerBlock cells -
     * depending on the grid contiguous dimension size,
     * that can range from a part of a single gridline to several complete gridlines.
     */
    const int threadLocalId     = threadIdx.x;
    const int gridLineSize      = localCountMinor;
    const int gridLineIndex     = threadLocalId / gridLineSize;
    const int gridLineCellIndex = threadLocalId - gridLineSize * gridLineIndex;
    const int gridLinesPerBlock = max(blockDim.x / gridLineSize, 1);
    const int activeWarps       = (blockDim.x / warpSize);
    const int indexMinor        = blockIdx.x * blockDim.x + gridLineCellIndex;
    const int indexMiddle       = blockIdx.y * gridLinesPerBlock + gridLineIndex;
    const int indexMajor        = blockIdx.z;

    /* Optional outputs */
    float energy = 0.0f;
    float virxx  = 0.0f;
    float virxy  = 0.0f;
    float virxz  = 0.0f;
    float viryy  = 0.0f;
    float viryz  = 0.0f;
    float virzz  = 0.0f;

    assert(indexMajor < kernelParams.grid.complexGridSize[majorDim]);
    if ((indexMiddle < localCountMiddle) & (indexMinor < localCountMinor)
        & (gridLineIndex < gridLinesPerBlock))
    {
        /* The offset should be equal to the global thread index for coalesced access */
        const int gridThreadIndex =
                (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;
        float2* __restrict__ gm_gridCell = gm_grid + gridThreadIndex;

        const int kMajor = indexMajor + localOffsetMajor;
        /* Checking either X in XYZ, or Y in YZX cases */
        const float mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int kMiddle = indexMiddle + localOffsetMiddle;
        float     mMiddle = kMiddle;
        /* Checking Y in XYZ case */
        if (gridOrdering == GridOrdering::XYZ)
        {
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);
        }
        const int kMinor = localOffsetMinor + indexMinor;
        float     mMinor = kMinor;
        /* Checking X in YZX case */
        if (gridOrdering == GridOrdering::YZX)
        {
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
        }
        /* We should skip the k-space point (0,0,0) */
        const bool notZeroPoint = (kMinor > 0) | (kMajor > 0) | (kMiddle > 0);

        float mX, mY, mZ;
        switch (gridOrdering)
        {
            case GridOrdering::YZX:
                mX = mMinor;
                mY = mMajor;
                mZ = mMiddle;
                break;

            case GridOrdering::XYZ:
                mX = mMajor;
                mY = mMiddle;
                mZ = mMinor;
                break;

            default: assert(false);
        }

        /* 0.5 correction factor for the first and last components of a Z dimension */
        float corner_fac = 1.0f;
        switch (gridOrdering)
        {
            case GridOrdering::YZX:
                if ((kMiddle == 0) | (kMiddle == maxkMiddle))
                {
                    corner_fac = 0.5f;
                }
                break;

            case GridOrdering::XYZ:
                if ((kMinor == 0) | (kMinor == maxkMinor))
                {
                    corner_fac = 0.5f;
                }
                break;

            default: assert(false);
        }

        if (notZeroPoint)
        {
            const float mhxk = mX * kernelParams.current.recipBox[XX][XX];
            const float mhyk = mX * kernelParams.current.recipBox[XX][YY]
                               + mY * kernelParams.current.recipBox[YY][YY];
            const float mhzk = mX * kernelParams.current.recipBox[XX][ZZ]
                               + mY * kernelParams.current.recipBox[YY][ZZ]
                               + mZ * kernelParams.current.recipBox[ZZ][ZZ];

            const float m2k = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            assert(m2k != 0.0f);
            // TODO: use LDG/textures for gm_splineValue
            float denom = m2k * float(HIP_PI_F) * kernelParams.current.boxVolume
                          * gm_splineValueMajor[kMajor] * gm_splineValueMiddle[kMiddle]
                          * gm_splineValueMinor[kMinor];
            assert(isfinite(denom));
            assert(denom != 0.0f);

            const float tmp1   = __expf(-kernelParams.grid.ewaldFactor * m2k);
            const float etermk = kernelParams.constants.elFactor * tmp1 / denom;

            float2       gridValue    = *gm_gridCell;
            const float2 oldGridValue = gridValue;
            gridValue.x *= etermk;
            gridValue.y *= etermk;
            *gm_gridCell = gridValue;

            if (computeEnergyAndVirial)
            {
                const float tmp1k =
                        2.0f * (gridValue.x * oldGridValue.x + gridValue.y * oldGridValue.y);

                float vfactor = (kernelParams.grid.ewaldFactor + 1.0f / m2k) * 2.0f;
                float ets2    = corner_fac * tmp1k;
                energy        = ets2;

                float ets2vf = ets2 * vfactor;

                virxx = ets2vf * mhxk * mhxk - ets2;
                virxy = ets2vf * mhxk * mhyk;
                virxz = ets2vf * mhxk * mhzk;
                viryy = ets2vf * mhyk * mhyk - ets2;
                viryz = ets2vf * mhyk * mhzk;
                virzz = ets2vf * mhzk * mhzk - ets2;
            }
        }
    }

    /* Optional energy/virial reduction */
    if (computeEnergyAndVirial)
    {
        virxx += warp_move_dpp<float, 0xb1>(virxx);
        viryy += warp_move_dpp<float, 0xb1>(viryy);
        virzz += warp_move_dpp<float, 0xb1>(virzz);
        virxy += warp_move_dpp<float, 0xb1>(virxy);
        virxz += warp_move_dpp<float, 0xb1>(virxz);
        viryz += warp_move_dpp<float, 0xb1>(viryz);
        energy += warp_move_dpp<float, 0xb1>(energy);

        virxx += warp_move_dpp<float, 0x4e>(virxx);
        viryy += warp_move_dpp<float, 0x4e>(viryy);
        virzz += warp_move_dpp<float, 0x4e>(virzz);
        virxy += warp_move_dpp<float, 0x4e>(virxy);
        virxz += warp_move_dpp<float, 0x4e>(virxz);
        viryz += warp_move_dpp<float, 0x4e>(viryz);
        energy += warp_move_dpp<float, 0x4e>(energy);

        virxx += warp_move_dpp<float, 0x114>(virxx);
        viryy += warp_move_dpp<float, 0x114>(viryy);
        virzz += warp_move_dpp<float, 0x114>(virzz);
        virxy += warp_move_dpp<float, 0x114>(virxy);
        virxz += warp_move_dpp<float, 0x114>(virxz);
        viryz += warp_move_dpp<float, 0x114>(viryz);
        energy += warp_move_dpp<float, 0x114>(energy);

        virxx += warp_move_dpp<float, 0x118>(virxx);
        viryy += warp_move_dpp<float, 0x118>(viryy);
        virzz += warp_move_dpp<float, 0x118>(virzz);
        virxy += warp_move_dpp<float, 0x118>(virxy);
        virxz += warp_move_dpp<float, 0x118>(virxz);
        viryz += warp_move_dpp<float, 0x118>(viryz);
        energy += warp_move_dpp<float, 0x118>(energy);

#ifndef __gfx1030__
        virxx += warp_move_dpp<float, 0x142>(virxx);
        viryy += warp_move_dpp<float, 0x142>(viryy);
        virzz += warp_move_dpp<float, 0x142>(virzz);
        virxy += warp_move_dpp<float, 0x142>(virxy);
        virxz += warp_move_dpp<float, 0x142>(virxz);
        viryz += warp_move_dpp<float, 0x142>(viryz);
        energy += warp_move_dpp<float, 0x142>(energy);
#else
        virxx += __shfl(virxx, 15, warpSize);
        viryy += __shfl(viryy, 15, warpSize);
        virzz += __shfl(virzz, 15, warpSize);
        virxy += __shfl(virxy, 15, warpSize);
        virxz += __shfl(virxz, 15, warpSize);
        viryz += __shfl(viryz, 15, warpSize);
        energy += __shfl(energy, 15, warpSize);
#endif

#ifndef __gfx1030__
        if (warpSize > 32)
        {
            virxx += warp_move_dpp<float, 0x143>(virxx);
            viryy += warp_move_dpp<float, 0x143>(viryy);
            virzz += warp_move_dpp<float, 0x143>(virzz);
            virxy += warp_move_dpp<float, 0x143>(virxy);
            virxz += warp_move_dpp<float, 0x143>(virxz);
            viryz += warp_move_dpp<float, 0x143>(viryz);
            energy += warp_move_dpp<float, 0x143>(energy);
        }
#endif
        const int componentIndex = threadLocalId & (warpSize - 1);
        __shared__ float sm_virialAndEnergy[c_virialAndEnergyCount][warpSize];

        if (componentIndex == (warpSize - 1))
        {
            const int warpIndex              = threadLocalId / warpSize;
            sm_virialAndEnergy[0][warpIndex] = virxx;
            sm_virialAndEnergy[1][warpIndex] = viryy;
            sm_virialAndEnergy[2][warpIndex] = virzz;
            sm_virialAndEnergy[3][warpIndex] = virxy;
            sm_virialAndEnergy[4][warpIndex] = virxz;
            sm_virialAndEnergy[5][warpIndex] = viryz;
            sm_virialAndEnergy[6][warpIndex] = energy;
        }
        __syncthreads();

        /* Now use shuffle again for each component */
        /* NOTE: This reduction assumes that activeWarps is a power of two
         */
         if (threadLocalId < activeWarps)
         {
             virxx = sm_virialAndEnergy[0][threadLocalId];
             viryy = sm_virialAndEnergy[1][threadLocalId];
             virzz = sm_virialAndEnergy[2][threadLocalId];
             virxy = sm_virialAndEnergy[3][threadLocalId];
             virxz = sm_virialAndEnergy[4][threadLocalId];
             viryz = sm_virialAndEnergy[5][threadLocalId];
             energy = sm_virialAndEnergy[6][threadLocalId];

             for (int offset = (activeWarps >> 1); offset > 0; offset >>= 1)
             {
                 virxx += __shfl_down(virxx, offset);
                 viryy += __shfl_down(viryy, offset);
                 virzz += __shfl_down(virzz, offset);
                 virxy += __shfl_down(virxy, offset);
                 virxz += __shfl_down(virxz, offset);
                 viryz += __shfl_down(viryz, offset);
                 energy += __shfl_down(energy, offset);
             }
             /* Final output */
             if (componentIndex == 0)
             {
                 atomicAddNoRet(gm_virialAndEnergy, virxx);
                 atomicAddNoRet(gm_virialAndEnergy + 1, viryy);
                 atomicAddNoRet(gm_virialAndEnergy + 2, virzz);
                 atomicAddNoRet(gm_virialAndEnergy + 3, virxy);
                 atomicAddNoRet(gm_virialAndEnergy + 4, virxz);
                 atomicAddNoRet(gm_virialAndEnergy + 5, viryz);
                 atomicAddNoRet(gm_virialAndEnergy + 6, energy);
             }
         }
    }
}

//! Kernel instantiations
template __global__ void pme_solve_kernel<GridOrdering::YZX, true, 0>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::YZX, false, 0>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::XYZ, true, 0>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::XYZ, false, 0>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::YZX, true, 1>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::YZX, false, 1>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::XYZ, true, 1>(const PmeGpuHipKernelParams);
template __global__ void pme_solve_kernel<GridOrdering::XYZ, false, 1>(const PmeGpuHipKernelParams);
