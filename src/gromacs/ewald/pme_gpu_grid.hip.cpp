/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
 * \brief Implements PME GPU halo exchange and PME GPU - Host FFT grid conversion
 * functions. These functions are used for PME decomposition in mixed-mode
 *
 * \author Gaurav Garg <gaugarg@nvidia.com>
 *
 * \ingroup module_ewald
 */

#include "gmxpre.h"

#include "pme_gpu_grid.h"

#include "config.h"

#include <cstdlib>

#include "gromacs/math/vec.h"
#include "gromacs/gpu_utils/hiputils.hpp"
#include "gromacs/gpu_utils/devicebuffer.hpp"
#include "pme.hpp"
#include "pme_gpu_types_host.h"
#include "pme_gpu_types.h"
#include "pme_gpu_types_host_impl.h"
#include "gromacs/fft/parallel_3dfft.h"

/*! \brief
 * A HIP kernel which packs non-contiguous overlap data in all 8 neighboring directions
 *
 */
static __global__ void pmeGpuPackHaloExternal(float* __restrict__ gm_realGrid,
                                              float* __restrict__ gm_transferGridUp,
                                              float* __restrict__ gm_transferGridDown,
                                              float* __restrict__ gm_transferGridLeft,
                                              float* __restrict__ gm_transferGridRight,
                                              float* __restrict__ gm_transferGridUpLeft,
                                              float* __restrict__ gm_transferGridDownLeft,
                                              float* __restrict__ gm_transferGridUpRight,
                                              float* __restrict__ gm_transferGridDownRight,
                                              int  overlapSizeUp,
                                              int  overlapSizeDown,
                                              int  overlapSizeLeft,
                                              int  overlapSizeRight,
                                              int  myGridX,
                                              int  myGridY,
                                              int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    // we might get iz greather than pmeSize.z when pmeSize.z is not
    // multiple of threadsAlongZDim(see below)
    if (iz >= pmeSize.z || iy >= myGridY)
    {
        return;
    }

    // up
    if (ix < overlapSizeUp)
    {
        int pmeIndex = (ix + pmeSize.x - overlapSizeUp) * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;
        int packedIndex                = ix * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridUp[packedIndex] = gm_realGrid[pmeIndex];
    }

    // down
    if (ix >= myGridX - overlapSizeDown)
    {
        int pmeIndex = (ix + overlapSizeDown) * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;
        int packedIndex = (ix - (myGridX - overlapSizeDown)) * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridDown[packedIndex] = gm_realGrid[pmeIndex];
    }

    // left
    if (iy < overlapSizeLeft)
    {
        int pmeIndex = ix * pmeSize.y * pmeSize.z + (iy + pmeSize.y - overlapSizeLeft) * pmeSize.z + iz;
        int packedIndex                  = ix * overlapSizeLeft * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridLeft[packedIndex] = gm_realGrid[pmeIndex];
    }

    // right
    if (iy >= myGridY - overlapSizeRight)
    {
        int pmeIndex    = ix * pmeSize.y * pmeSize.z + (iy + overlapSizeRight) * pmeSize.z + iz;
        int packedIndex = ix * overlapSizeRight * pmeSize.z
                          + (iy - (myGridY - overlapSizeRight)) * pmeSize.z + iz;
        gm_transferGridRight[packedIndex] = gm_realGrid[pmeIndex];
    }

    // up left
    if (ix < overlapSizeUp && iy < overlapSizeLeft)
    {
        int pmeIndex = (ix + pmeSize.x - overlapSizeUp) * pmeSize.y * pmeSize.z
                       + (iy + pmeSize.y - overlapSizeLeft) * pmeSize.z + iz;
        int packedIndex                    = ix * overlapSizeLeft * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridUpLeft[packedIndex] = gm_realGrid[pmeIndex];
    }

    // down left
    if (ix >= myGridX - overlapSizeDown && iy < overlapSizeLeft)
    {
        int pmeIndex = (ix + overlapSizeDown) * pmeSize.y * pmeSize.z
                       + (iy + pmeSize.y - overlapSizeLeft) * pmeSize.z + iz;
        int packedIndex =
                (ix - (myGridX - overlapSizeDown)) * overlapSizeLeft * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridDownLeft[packedIndex] = gm_realGrid[pmeIndex];
    }

    // up right
    if (ix < overlapSizeUp && iy >= myGridY - overlapSizeRight)
    {
        int pmeIndex = (ix + pmeSize.x - overlapSizeUp) * pmeSize.y * pmeSize.z
                       + (iy + overlapSizeRight) * pmeSize.z + iz;
        int packedIndex = ix * overlapSizeRight * pmeSize.z
                          + (iy - (myGridY - overlapSizeRight)) * pmeSize.z + iz;
        gm_transferGridUpRight[packedIndex] = gm_realGrid[pmeIndex];
    }

    // down right
    if (ix >= myGridX - overlapSizeDown && iy >= myGridY - overlapSizeRight)
    {
        int pmeIndex = (ix + overlapSizeDown) * pmeSize.y * pmeSize.z
                       + (iy + overlapSizeRight) * pmeSize.z + iz;
        int packedIndex = (ix - (myGridX - overlapSizeDown)) * overlapSizeRight * pmeSize.z
                          + (iy - (myGridY - overlapSizeRight)) * pmeSize.z + iz;
        gm_transferGridDownRight[packedIndex] = gm_realGrid[pmeIndex];
    }
}

/*! \brief
 * A HIP kernel which assigns data in halo region in all 8 neighboring directios
 */
static __global__ void pmeGpuAssignHaloExternal(float* __restrict__ gm_realGrid,
                                                float* __restrict__ gm_transferGridUp,
                                                float* __restrict__ gm_transferGridDown,
                                                float* __restrict__ gm_transferGridLeft,
                                                float* __restrict__ gm_transferGridRight,
                                                float* __restrict__ gm_transferGridUpLeft,
                                                float* __restrict__ gm_transferGridDownLeft,
                                                float* __restrict__ gm_transferGridUpRight,
                                                float* __restrict__ gm_transferGridDownRight,
                                                int  overlapSizeUp,
                                                int  overlapSizeDown,
                                                int  overlapSizeLeft,
                                                int  overlapSizeRight,
                                                int  myGridX,
                                                int  myGridY,
                                                int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    // we might get iz greather than pmeSize.z when pmeSize.z is not
    // multiple of threadsAlongZDim(see below)
    if (iz >= pmeSize.z || iy >= myGridY)
    {
        return;
    }

    // up
    if (ix < overlapSizeUp)
    {
        int pmeIndex = (ix + pmeSize.x - overlapSizeUp) * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;
        int packedIndex       = ix * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridUp[packedIndex];
    }

    // down
    if (ix >= myGridX - overlapSizeDown)
    {
        int pmeIndex = (ix + overlapSizeDown) * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;
        int packedIndex = (ix - (myGridX - overlapSizeDown)) * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridDown[packedIndex];
    }

    // left
    if (iy < overlapSizeLeft)
    {
        int pmeIndex = ix * pmeSize.y * pmeSize.z + (iy + pmeSize.y - overlapSizeLeft) * pmeSize.z + iz;
        int packedIndex       = ix * overlapSizeLeft * pmeSize.z + iy * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridLeft[packedIndex];
    }

    // right
    if (iy >= myGridY - overlapSizeRight)
    {
        int pmeIndex    = ix * pmeSize.y * pmeSize.z + (iy + overlapSizeRight) * pmeSize.z + iz;
        int packedIndex = ix * overlapSizeRight * pmeSize.z
                          + (iy - (myGridY - overlapSizeRight)) * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridRight[packedIndex];
    }

    // up left
    if (ix < overlapSizeUp && iy < overlapSizeLeft)
    {
        int pmeIndex = (ix + pmeSize.x - overlapSizeUp) * pmeSize.y * pmeSize.z
                       + (iy + pmeSize.y - overlapSizeLeft) * pmeSize.z + iz;
        int packedIndex       = ix * overlapSizeLeft * pmeSize.z + iy * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridUpLeft[packedIndex];
    }

    // down left
    if (ix >= myGridX - overlapSizeDown && iy < overlapSizeLeft)
    {
        int pmeIndex = (ix + overlapSizeDown) * pmeSize.y * pmeSize.z
                       + (iy + pmeSize.y - overlapSizeLeft) * pmeSize.z + iz;
        int packedIndex =
                (ix - (myGridX - overlapSizeDown)) * overlapSizeLeft * pmeSize.z + iy * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridDownLeft[packedIndex];
    }

    // up right
    if (ix < overlapSizeUp && iy >= myGridY - overlapSizeRight)
    {
        int pmeIndex = (ix + pmeSize.x - overlapSizeUp) * pmeSize.y * pmeSize.z
                       + (iy + overlapSizeRight) * pmeSize.z + iz;
        int packedIndex = ix * overlapSizeRight * pmeSize.z
                          + (iy - (myGridY - overlapSizeRight)) * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridUpRight[packedIndex];
    }

    // down right
    if (ix >= myGridX - overlapSizeDown && iy >= myGridY - overlapSizeRight)
    {
        int pmeIndex = (ix + overlapSizeDown) * pmeSize.y * pmeSize.z
                       + (iy + overlapSizeRight) * pmeSize.z + iz;
        int packedIndex = (ix - (myGridX - overlapSizeDown)) * overlapSizeRight * pmeSize.z
                          + (iy - (myGridY - overlapSizeRight)) * pmeSize.z + iz;
        gm_realGrid[pmeIndex] = gm_transferGridDownRight[packedIndex];
    }
}

/*! \brief
 * A HIP kernel which adds grid overlap data received from neighboring ranks
 */

static __global__ void pmeGpuAddHaloInternal(float* __restrict__ gm_realGrid,
                                             float* __restrict__ gm_transferGridUp,
                                             float* __restrict__ gm_transferGridDown,
                                             float* __restrict__ gm_transferGridLeft,
                                             float* __restrict__ gm_transferGridRight,
                                             float* __restrict__ gm_transferGridUpLeft,
                                             float* __restrict__ gm_transferGridDownLeft,
                                             float* __restrict__ gm_transferGridUpRight,
                                             float* __restrict__ gm_transferGridDownRight,
                                             int  overlapSizeX,
                                             int  overlapSizeY,
                                             int  overlapUp,
                                             int  overlapLeft,
                                             int  myGridX,
                                             int  myGridY,
                                             int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    // we might get iz greather than pmeSize.z when pmeSize.z is not
    // multiple of threadsAlongZDim(see below)
    if (iz >= pmeSize.z || iy >= myGridY)
    {
        return;
    }

    int pmeIndex = ix * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;

    float val = gm_realGrid[pmeIndex];

    // up rank
    if (ix < overlapSizeX)
    {
        int packedIndex = ix * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        val += gm_transferGridUp[packedIndex];
    }

    // down rank
    if (ix >= myGridX - overlapSizeX && overlapUp > 0)
    {
        int packedIndex = (ix - (myGridX - overlapSizeX)) * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        val += gm_transferGridDown[packedIndex];
    }

    // left rank
    if (iy < overlapSizeY)
    {
        int packedIndex = ix * overlapSizeY * pmeSize.z + iy * pmeSize.z + iz;
        val += gm_transferGridLeft[packedIndex];
    }

    // right rank
    if (iy >= myGridY - overlapSizeY && overlapLeft > 0)
    {
        int packedIndex = ix * overlapSizeY * pmeSize.z + (iy - (myGridY - overlapSizeY)) * pmeSize.z + iz;
        val += gm_transferGridRight[packedIndex];
    }

    // up left rank
    if (ix < overlapSizeX && iy < overlapSizeY)
    {
        int packedIndex = ix * overlapSizeY * pmeSize.z + iy * pmeSize.z + iz;
        val += gm_transferGridUpLeft[packedIndex];
    }

    // up right rank
    if (ix < overlapSizeX && iy >= myGridY - overlapSizeY && overlapLeft > 0)
    {
        int packedIndex = ix * overlapSizeY * pmeSize.z + (iy - (myGridY - overlapSizeY)) * pmeSize.z + iz;
        val += gm_transferGridUpRight[packedIndex];
    }

    // down left rank
    if (ix >= myGridX - overlapSizeX && overlapUp > 0 && iy < overlapSizeY)
    {
        int packedIndex = (ix - (myGridX - overlapSizeX)) * overlapSizeY * pmeSize.z + iy * pmeSize.z + iz;
        val += gm_transferGridDownLeft[packedIndex];
    }

    // down right rank
    if (ix >= myGridX - overlapSizeX && overlapUp > 0 && iy >= myGridY - overlapSizeY && overlapLeft > 0)
    {
        int packedIndex = (ix - (myGridX - overlapSizeX)) * overlapSizeY * pmeSize.z
                          + (iy - (myGridY - overlapSizeY)) * pmeSize.z + iz;
        val += gm_transferGridDownRight[packedIndex];
    }

    gm_realGrid[pmeIndex] = val;
}

/*! \brief
 * A HIP kernel which packs non-contiguous overlap data in all 8 neighboring directions
 *
 */
static __global__ void pmeGpuPackHaloInternal(float* __restrict__ gm_realGrid,
                                              float* __restrict__ gm_transferGridUp,
                                              float* __restrict__ gm_transferGridDown,
                                              float* __restrict__ gm_transferGridLeft,
                                              float* __restrict__ gm_transferGridRight,
                                              float* __restrict__ gm_transferGridUpLeft,
                                              float* __restrict__ gm_transferGridDownLeft,
                                              float* __restrict__ gm_transferGridUpRight,
                                              float* __restrict__ gm_transferGridDownRight,
                                              int  overlapSizeX,
                                              int  overlapSizeY,
                                              int  overlapUp,
                                              int  overlapLeft,
                                              int  myGridX,
                                              int  myGridY,
                                              int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    // we might get iz greather than pmeSize.z when pmeSize.z is not
    // multiple of threadsAlongZDim(see below)
    if (iz >= pmeSize.z || iy >= myGridY)
    {
        return;
    }

    int pmeIndex = ix * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;

    float val = gm_realGrid[pmeIndex];

    // up rank
    if (ix < overlapSizeX)
    {
        int packedIndex                = ix * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridUp[packedIndex] = val;
    }

    // down rank
    if (ix >= myGridX - overlapSizeX && overlapUp > 0)
    {
        int packedIndex = (ix - (myGridX - overlapSizeX)) * myGridY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridDown[packedIndex] = val;
    }

    // left rank
    if (iy < overlapSizeY)
    {
        int packedIndex                  = ix * overlapSizeY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridLeft[packedIndex] = val;
    }

    // right rank
    if (iy >= myGridY - overlapSizeY && overlapLeft > 0)
    {
        int packedIndex = ix * overlapSizeY * pmeSize.z + (iy - (myGridY - overlapSizeY)) * pmeSize.z + iz;
        gm_transferGridRight[packedIndex] = val;
    }

    // up left rank
    if (ix < overlapSizeX && iy < overlapSizeY)
    {
        int packedIndex                    = ix * overlapSizeY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridUpLeft[packedIndex] = val;
    }

    // down left rank
    if (ix >= myGridX - overlapSizeX && overlapUp > 0 && iy < overlapSizeY)
    {
        int packedIndex = (ix - (myGridX - overlapSizeX)) * overlapSizeY * pmeSize.z + iy * pmeSize.z + iz;
        gm_transferGridDownLeft[packedIndex] = val;
    }

    // up right rank
    if (ix < overlapSizeX && iy >= myGridY - overlapSizeY && overlapLeft > 0)
    {
        int packedIndex = ix * overlapSizeY * pmeSize.z + (iy - (myGridY - overlapSizeY)) * pmeSize.z + iz;
        gm_transferGridUpRight[packedIndex] = val;
    }

    // down right rank
    if (ix >= myGridX - overlapSizeX && overlapUp > 0 && iy >= myGridY - overlapSizeY && overlapLeft > 0)
    {
        int packedIndex = (ix - (myGridX - overlapSizeX)) * overlapSizeY * pmeSize.z
                          + (iy - (myGridY - overlapSizeY)) * pmeSize.z + iz;
        gm_transferGridDownRight[packedIndex] = val;
    }
}

/*! \brief
 * A HIP kernel which copies data from pme grid to FFT grid and back
 *
 * \param[in] gm_pmeGrid          local PME grid
 * \param[in] gm_fftGrid          local FFT grid
 * \param[in] fftNData           local FFT grid size without padding
 * \param[in] fftSize            local FFT grid padded size
 * \param[in] pmeSize            local PME grid padded size
 *
 * \tparam  pmeToFft               A boolean which tells if this is conversion from PME grid to FFT grid or reverse
 */
template<bool pmeToFft>
static __global__ void pmegrid_to_fftgrid(float* __restrict__ gm_realGrid,
                                          float* __restrict__ gm_fftGrid,
                                          int3 fftNData,
                                          int3 fftSize,
                                          int3 pmeSize)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int ix = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= fftNData.x || iy >= fftNData.y || iz >= fftNData.z)
    {
        return;
    }

    int fftidx   = ix * fftSize.y * fftSize.z + iy * fftSize.z + iz;
    int pmeIndex = ix * pmeSize.y * pmeSize.z + iy * pmeSize.z + iz;

    if (pmeToFft)
    {
        gm_fftGrid[fftidx] = gm_realGrid[pmeIndex];
    }
    else
    {
        gm_realGrid[pmeIndex] = gm_fftGrid[fftidx];
    }
}

/*! \brief
 * Launches HIP kernel to pack non-contiguous external halo data
 */
static void packHaloDataExternal(const PmeGpu* pmeGpu,
                                 int           overlapUp,
                                 int           overlapDown,
                                 int           overlapLeft,
                                 int           overlapRight,
                                 int           myGridX,
                                 int           myGridY,
                                 const ivec&   pmeSize,
                                 float*        realGrid,
                                 float*        packedGridUp,
                                 float*        packedGridDown,
                                 float*        packedGridLeft,
                                 float*        packedGridRight,
                                 float*        packedGridUpLeft,
                                 float*        packedGridDownLeft,
                                 float*        packedGridUpRight,
                                 float*        packedGridDownRight)
{
    // keeping same as warp size for better coalescing
    // Not keeping to higher value such as 64 to avoid high masked out
    // inactive threads as FFT grid sizes tend to be quite small
    const int threadsAlongZDim = 32;
    const int threadsAlongYDim = 4;

    // right grid
    KernelLaunchConfig config;
    config.blockSize[0]     = threadsAlongZDim;
    config.blockSize[1]     = threadsAlongYDim;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (pmeSize[ZZ] + threadsAlongZDim - 1) / threadsAlongZDim;
    config.gridSize[1]      = (myGridY + threadsAlongYDim - 1) / threadsAlongYDim;
    config.gridSize[2]      = myGridX;
    config.sharedMemorySize = 0;

    auto kernelFn   = pmeGpuPackHaloExternal;
    auto kernelArgs = prepareGpuKernelArguments(kernelFn,
                                                config,
                                                &realGrid,
                                                &packedGridUp,
                                                &packedGridDown,
                                                &packedGridLeft,
                                                &packedGridRight,
                                                &packedGridUpLeft,
                                                &packedGridDownLeft,
                                                &packedGridUpRight,
                                                &packedGridDownRight,
                                                &overlapUp,
                                                &overlapDown,
                                                &overlapLeft,
                                                &overlapRight,
                                                &myGridX,
                                                &myGridY,
                                                &pmeSize);

    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Pack Grid Halo Exchange",
                    kernelArgs);
}

/*! \brief
 * Launches HIP kernel to pack non-contiguous internal halo data
 */
static void packHaloDataInternal(const PmeGpu* pmeGpu,
                                 int           overlapSizeX,
                                 int           overlapSizeY,
                                 int           overlapUp,
                                 int           overlapLeft,
                                 int           myGridX,
                                 int           myGridY,
                                 const ivec&   pmeSize,
                                 float*        realGrid,
                                 float*        packedGridUp,
                                 float*        packedGridDown,
                                 float*        packedGridLeft,
                                 float*        packedGridRight,
                                 float*        packedGridUpLeft,
                                 float*        packedGridDownLeft,
                                 float*        packedGridUpRight,
                                 float*        packedGridDownRight)
{
    // keeping same as warp size for better coalescing
    // Not keeping to higher value such as 64 to avoid high masked out
    // inactive threads as FFT grid sizes tend to be quite small
    const int threadsAlongZDim = 32;
    const int threadsAlongYDim = 4;

    // right grid
    KernelLaunchConfig config;
    config.blockSize[0]     = threadsAlongZDim;
    config.blockSize[1]     = threadsAlongYDim;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (pmeSize[ZZ] + threadsAlongZDim - 1) / threadsAlongZDim;
    config.gridSize[1]      = (myGridY + threadsAlongYDim - 1) / threadsAlongYDim;
    config.gridSize[2]      = myGridX;
    config.sharedMemorySize = 0;

    auto kernelFn   = pmeGpuPackHaloInternal;
    auto kernelArgs = prepareGpuKernelArguments(kernelFn,
                                                config,
                                                &realGrid,
                                                &packedGridUp,
                                                &packedGridDown,
                                                &packedGridLeft,
                                                &packedGridRight,
                                                &packedGridUpLeft,
                                                &packedGridDownLeft,
                                                &packedGridUpRight,
                                                &packedGridDownRight,
                                                &overlapSizeX,
                                                &overlapSizeY,
                                                &overlapUp,
                                                &overlapLeft,
                                                &myGridX,
                                                &myGridY,
                                                &pmeSize);

    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Pack Grid Halo Exchange",
                    kernelArgs);
}


/*! \brief
 * Launches HIP kernel to reduce overlap data
 */
static void reduceHaloData(const PmeGpu* pmeGpu,
                           int           overlapSizeX,
                           int           overlapSizeY,
                           int           overlapUp,
                           int           overlapLeft,
                           int           myGridX,
                           int           myGridY,
                           const ivec&   pmeSize,
                           float*        realGrid,
                           float*        packedGridUp,
                           float*        packedGridDown,
                           float*        packedGridLeft,
                           float*        packedGridRight,
                           float*        packedGridUpLeft,
                           float*        packedGridDownLeft,
                           float*        packedGridUpRight,
                           float*        packedGridDownRight)
{
    // keeping same as warp size for better coalescing
    // Not keeping to higher value such as 64 to avoid high masked out
    // inactive threads as FFT grid sizes tend to be quite small
    const int threadsAlongZDim = 32;
    const int threadsAlongYDim = 4;

    // right grid
    KernelLaunchConfig config;
    config.blockSize[0]     = threadsAlongZDim;
    config.blockSize[1]     = threadsAlongYDim;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (pmeSize[ZZ] + threadsAlongZDim - 1) / threadsAlongZDim;
    config.gridSize[1]      = (myGridY + threadsAlongYDim - 1) / threadsAlongYDim;
    config.gridSize[2]      = myGridX;
    config.sharedMemorySize = 0;

    auto kernelFn = pmeGpuAddHaloInternal;

    auto kernelArgs = prepareGpuKernelArguments(kernelFn,
                                                config,
                                                &realGrid,
                                                &packedGridUp,
                                                &packedGridDown,
                                                &packedGridLeft,
                                                &packedGridRight,
                                                &packedGridUpLeft,
                                                &packedGridDownLeft,
                                                &packedGridUpRight,
                                                &packedGridDownRight,
                                                &overlapSizeX,
                                                &overlapSizeY,
                                                &overlapUp,
                                                &overlapLeft,
                                                &myGridX,
                                                &myGridY,
                                                &pmeSize);


    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Pack Grid Halo Exchange",
                    kernelArgs);
}

/*! \brief
 * Launches HIP kernel to initialize overlap data
 */
static void assignHaloData(const PmeGpu* pmeGpu,
                           int           overlapUp,
                           int           overlapDown,
                           int           overlapLeft,
                           int           overlapRight,
                           int           myGridX,
                           int           myGridY,
                           const ivec&   pmeSize,
                           float*        realGrid,
                           float*        packedGridUp,
                           float*        packedGridDown,
                           float*        packedGridLeft,
                           float*        packedGridRight,
                           float*        packedGridUpLeft,
                           float*        packedGridDownLeft,
                           float*        packedGridUpRight,
                           float*        packedGridDownRight)
{
    // keeping same as warp size for better coalescing
    // Not keeping to higher value such as 64 to avoid high masked out
    // inactive threads as FFT grid sizes tend to be quite small
    const int threadsAlongZDim = 32;
    const int threadsAlongYDim = 4;

    // right grid
    KernelLaunchConfig config;
    config.blockSize[0]     = threadsAlongZDim;
    config.blockSize[1]     = threadsAlongYDim;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (pmeSize[ZZ] + threadsAlongZDim - 1) / threadsAlongZDim;
    config.gridSize[1]      = (myGridY + threadsAlongYDim - 1) / threadsAlongYDim;
    config.gridSize[2]      = myGridX;
    config.sharedMemorySize = 0;

    auto kernelFn   = pmeGpuAssignHaloExternal;
    auto kernelArgs = prepareGpuKernelArguments(kernelFn,
                                                config,
                                                &realGrid,
                                                &packedGridUp,
                                                &packedGridDown,
                                                &packedGridLeft,
                                                &packedGridRight,
                                                &packedGridUpLeft,
                                                &packedGridDownLeft,
                                                &packedGridUpRight,
                                                &packedGridDownRight,
                                                &overlapUp,
                                                &overlapDown,
                                                &overlapLeft,
                                                &overlapRight,
                                                &myGridX,
                                                &myGridY,
                                                &pmeSize);


    launchGpuKernel(kernelFn,
                    config,
                    pmeGpu->archSpecific->pmeStream_,
                    nullptr,
                    "PME Domdec GPU Pack Grid Halo Exchange",
                    kernelArgs);
}

void pmeGpuGridHaloExchange(const PmeGpu* pmeGpu)
{
#if GMX_MPI
    // Note here we are assuming that width of the chunks is not so small that we need to
    // transfer to/from multiple ranks i.e. that the distributed grid contains chunks at least order-1 points wide.

    auto* kernelParamsPtr = pmeGpu->kernelParams.get();
    ivec  localPmeSize;
    localPmeSize[XX] = kernelParamsPtr->grid.realGridSizePadded[XX];
    localPmeSize[YY] = kernelParamsPtr->grid.realGridSizePadded[YY];
    localPmeSize[ZZ] = kernelParamsPtr->grid.realGridSizePadded[ZZ];

    int overlapSize = pmeGpu->common->gridHalo;

    int overlapX = 0;
    int overlapY = 0;

    int overlapDown = 0;
    int overlapUp   = 0;

    int overlapRight = 0;
    int overlapLeft  = 0;

    int rankX   = pmeGpu->common->nodeidX;
    int rankY   = pmeGpu->common->nodeidY;
    int myGridX = pmeGpu->common->s2g0X[rankX + 1] - pmeGpu->common->s2g0X[rankX];
    int myGridY = pmeGpu->common->s2g0Y[rankY + 1] - pmeGpu->common->s2g0Y[rankY];

    int sizeX = pmeGpu->common->nnodesX;
    int down  = (rankX + 1) % sizeX;
    int up    = (rankX + sizeX - 1) % sizeX;

    int sizeY = pmeGpu->common->nnodesY;
    int right = (rankY + 1) % sizeY;
    int left  = (rankY + sizeY - 1) % sizeY;

    // major dimension
    if (sizeX > 1)
    {
        // Note that s2g0[size] is the grid size (array is allocated to size+1)
        int downGrid = pmeGpu->common->s2g0X[down + 1] - pmeGpu->common->s2g0X[down];
        int upGrid   = pmeGpu->common->s2g0X[up + 1] - pmeGpu->common->s2g0X[up];

        // current implementation transfers from/to only immediate neighbours
        GMX_ASSERT(overlapSize <= myGridX && overlapSize <= downGrid && overlapSize <= upGrid,
                   "Exchange supported only with immediate neighbor");

        overlapX    = std::min(overlapSize, myGridX);
        overlapDown = std::min(overlapSize, downGrid);
        overlapUp   = std::min(overlapSize, upGrid);

        // if only 2 PME ranks in X-domain and overlap width more than slab width
        // just transfer all grid points from neighbor
        if (down == up && overlapDown + overlapUp >= downGrid)
        {
            overlapX    = myGridX;
            overlapDown = downGrid;
            overlapUp   = 0;
        }
    }

    // minor dimension
    if (sizeY > 1)
    {
        // Note that s2g0[size] is the grid size (array is allocated to size+1)
        int rightGrid = pmeGpu->common->s2g0Y[right + 1] - pmeGpu->common->s2g0Y[right];
        int leftGrid  = pmeGpu->common->s2g0Y[left + 1] - pmeGpu->common->s2g0Y[left];

        // current implementation transfers from/to only immediate neighbours
        GMX_ASSERT(overlapSize <= myGridY && overlapSize <= rightGrid && overlapSize <= leftGrid,
                   "Exchange supported only with immediate neighbor");

        overlapY     = std::min(overlapSize, myGridY);
        overlapRight = std::min(overlapSize, rightGrid);
        overlapLeft  = std::min(overlapSize, leftGrid);

        // if only 2 PME ranks in Y-domain and overlap width more than slab width
        // just transfer all grid points from neighbor
        if (right == left && overlapRight + overlapLeft >= rightGrid)
        {
            overlapY     = myGridY;
            overlapRight = rightGrid;
            overlapLeft  = 0;
        }
    }

    for (int gridIndex = 0; gridIndex < pmeGpu->common->ngrids; gridIndex++)
    {
        MPI_Request req[16];
        int         count    = 0;
        float*      realGrid = pmeGpu->kernelParams->grid.d_realGrid[gridIndex];

        float* sendGridUp   = pmeGpu->archSpecific->d_sendGridUp;
        float* sendGridDown = pmeGpu->archSpecific->d_sendGridDown;

        // no need to pack if slab-decomposition in X-dimension as data is already contiguous
        if (pmeGpu->common->nnodesY == 1)
        {
            int sendOffsetDown = myGridX * localPmeSize[YY] * localPmeSize[ZZ];
            int sendOffsetUp = (localPmeSize[XX] - overlapUp) * localPmeSize[YY] * localPmeSize[ZZ];
            sendGridUp       = &realGrid[sendOffsetUp];
            sendGridDown     = &realGrid[sendOffsetDown];
        }
        else
        {
            // launch packing kernel
            packHaloDataExternal(pmeGpu,
                                 overlapUp,
                                 overlapDown,
                                 overlapLeft,
                                 overlapRight,
                                 myGridX,
                                 myGridY,
                                 localPmeSize,
                                 realGrid,
                                 sendGridUp,
                                 sendGridDown,
                                 pmeGpu->archSpecific->d_sendGridLeft,
                                 pmeGpu->archSpecific->d_sendGridRight,
                                 pmeGpu->archSpecific->d_sendGridUpLeft,
                                 pmeGpu->archSpecific->d_sendGridDownLeft,
                                 pmeGpu->archSpecific->d_sendGridUpRight,
                                 pmeGpu->archSpecific->d_sendGridDownRight);
        }

        // make sure data is ready on GPU before MPI communication
        pmeGpu->archSpecific->pmeStream_.synchronize();


        // major dimension
        if (sizeX > 1)
        {
            constexpr int mpiTag = 403; // Arbitrarily chosen

            // send data to down rank and recv from up rank
            MPI_Irecv(pmeGpu->archSpecific->d_recvGridUp,
                      overlapX * myGridY * localPmeSize[ZZ],
                      MPI_FLOAT,
                      up,
                      mpiTag,
                      pmeGpu->common->mpiCommX,
                      &req[count++]);

            MPI_Isend(sendGridDown,
                      overlapDown * myGridY * localPmeSize[ZZ],
                      MPI_FLOAT,
                      down,
                      mpiTag,
                      pmeGpu->common->mpiCommX,
                      &req[count++]);

            if (overlapUp > 0)
            {
                // send data to up rank and recv from down rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridDown,
                          overlapX * myGridY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          down,
                          mpiTag,
                          pmeGpu->common->mpiCommX,
                          &req[count++]);

                MPI_Isend(sendGridUp,
                          overlapUp * myGridY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          up,
                          mpiTag,
                          pmeGpu->common->mpiCommX,
                          &req[count++]);
            }
        }

        // minor dimension
        if (sizeY > 1)
        {
            constexpr int mpiTag = 403; // Arbitrarily chosen

            // recv from left rank
            MPI_Irecv(pmeGpu->archSpecific->d_recvGridLeft,
                      overlapY * myGridX * localPmeSize[ZZ],
                      MPI_FLOAT,
                      left,
                      mpiTag,
                      pmeGpu->common->mpiCommY,
                      &req[count++]);

            // send data to right rank
            MPI_Isend(pmeGpu->archSpecific->d_sendGridRight,
                      overlapRight * myGridX * localPmeSize[ZZ],
                      MPI_FLOAT,
                      right,
                      mpiTag,
                      pmeGpu->common->mpiCommY,
                      &req[count++]);

            if (overlapLeft > 0)
            {
                // recv from right rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridRight,
                          overlapY * myGridX * localPmeSize[ZZ],
                          MPI_FLOAT,
                          right,
                          mpiTag,
                          pmeGpu->common->mpiCommY,
                          &req[count++]);

                // send data to left rank
                MPI_Isend(pmeGpu->archSpecific->d_sendGridLeft,
                          overlapLeft * myGridX * localPmeSize[ZZ],
                          MPI_FLOAT,
                          left,
                          mpiTag,
                          pmeGpu->common->mpiCommY,
                          &req[count++]);
            }
        }

        if (sizeX > 1 && sizeY > 1)
        {
            int rankUpLeft   = up * sizeY + left;
            int rankDownLeft = down * sizeY + left;

            int rankUpRight   = up * sizeY + right;
            int rankDownRight = down * sizeY + right;

            constexpr int mpiTag = 403; // Arbitrarily chosen

            // send data to down rank and recv from up rank
            MPI_Irecv(pmeGpu->archSpecific->d_recvGridUpLeft,
                      overlapX * overlapY * localPmeSize[ZZ],
                      MPI_FLOAT,
                      rankUpLeft,
                      mpiTag,
                      pmeGpu->common->mpiComm,
                      &req[count++]);

            MPI_Isend(pmeGpu->archSpecific->d_sendGridDownRight,
                      overlapDown * overlapRight * localPmeSize[ZZ],
                      MPI_FLOAT,
                      rankDownRight,
                      mpiTag,
                      pmeGpu->common->mpiComm,
                      &req[count++]);

            if (overlapLeft > 0)
            {
                // send data to up rank and recv from down rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridUpRight,
                          overlapX * overlapY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankUpRight,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);

                MPI_Isend(pmeGpu->archSpecific->d_sendGridDownLeft,
                          overlapDown * overlapLeft * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankDownLeft,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);
            }

            if (overlapUp > 0)
            {
                // send data to up rank and recv from down rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridDownLeft,
                          overlapX * overlapY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankDownLeft,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);

                MPI_Isend(pmeGpu->archSpecific->d_sendGridUpRight,
                          overlapUp * overlapRight * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankUpRight,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);
            }

            if (overlapUp > 0 && overlapLeft > 0)
            {
                // send data to up rank and recv from down rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridDownRight,
                          overlapX * overlapY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankDownRight,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);

                MPI_Isend(pmeGpu->archSpecific->d_sendGridUpLeft,
                          overlapUp * overlapLeft * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankUpLeft,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);
            }
        }

        MPI_Waitall(count, req, MPI_STATUSES_IGNORE);

        // reduce halo data
        reduceHaloData(pmeGpu,
                       overlapX,
                       overlapY,
                       overlapUp,
                       overlapLeft,
                       myGridX,
                       myGridY,
                       localPmeSize,
                       realGrid,
                       pmeGpu->archSpecific->d_recvGridUp,
                       pmeGpu->archSpecific->d_recvGridDown,
                       pmeGpu->archSpecific->d_recvGridLeft,
                       pmeGpu->archSpecific->d_recvGridRight,
                       pmeGpu->archSpecific->d_recvGridUpLeft,
                       pmeGpu->archSpecific->d_recvGridDownLeft,
                       pmeGpu->archSpecific->d_recvGridUpRight,
                       pmeGpu->archSpecific->d_recvGridDownRight);
    }
#else
    GMX_UNUSED_VALUE(pmeGpu);
#endif
}

void pmeGpuGridHaloExchangeReverse(const PmeGpu* pmeGpu)
{
#if GMX_MPI
    // Note here we are assuming that width of the chunks is not so small that we need to
    // transfer to/from multiple ranks i.e. that the distributed grid contains chunks at least order-1 points wide.

    auto* kernelParamsPtr = pmeGpu->kernelParams.get();
    ivec  localPmeSize;
    localPmeSize[XX] = kernelParamsPtr->grid.realGridSizePadded[XX];
    localPmeSize[YY] = kernelParamsPtr->grid.realGridSizePadded[YY];
    localPmeSize[ZZ] = kernelParamsPtr->grid.realGridSizePadded[ZZ];

    int overlapSize = pmeGpu->common->gridHalo;

    int overlapX = 0;
    int overlapY = 0;

    int overlapDown = 0;
    int overlapUp   = 0;

    int overlapRight = 0;
    int overlapLeft  = 0;

    int rankX   = pmeGpu->common->nodeidX;
    int rankY   = pmeGpu->common->nodeidY;
    int myGridX = pmeGpu->common->s2g0X[rankX + 1] - pmeGpu->common->s2g0X[rankX];
    int myGridY = pmeGpu->common->s2g0Y[rankY + 1] - pmeGpu->common->s2g0Y[rankY];

    int sizeX = pmeGpu->common->nnodesX;
    int down  = (rankX + 1) % sizeX;
    int up    = (rankX + sizeX - 1) % sizeX;

    int sizeY = pmeGpu->common->nnodesY;
    int right = (rankY + 1) % sizeY;
    int left  = (rankY + sizeY - 1) % sizeY;

    // major dimension
    if (sizeX > 1)
    {
        // Note that s2g0[size] is the grid size (array is allocated to size+1)
        int downGrid = pmeGpu->common->s2g0X[down + 1] - pmeGpu->common->s2g0X[down];
        int upGrid   = pmeGpu->common->s2g0X[up + 1] - pmeGpu->common->s2g0X[up];

        // current implementation transfers from/to only immediate neighbours
        GMX_ASSERT(overlapSize <= myGridX && overlapSize <= downGrid && overlapSize <= upGrid,
                   "Exchange supported only with immediate neighbor");

        overlapX    = std::min(overlapSize, myGridX);
        overlapDown = std::min(overlapSize, downGrid);
        overlapUp   = std::min(overlapSize, upGrid);

        // if only 2 PME ranks in X-domain and overlap width more than slab width
        // just transfer all grid points from neighbor
        if (down == up && overlapDown + overlapUp >= downGrid)
        {
            overlapX    = myGridX;
            overlapDown = downGrid;
            overlapUp   = 0;
        }
    }

    // minor dimension
    if (sizeY > 1)
    {
        // Note that s2g0[size] is the grid size (array is allocated to size+1)
        int rightGrid = pmeGpu->common->s2g0Y[right + 1] - pmeGpu->common->s2g0Y[right];
        int leftGrid  = pmeGpu->common->s2g0Y[left + 1] - pmeGpu->common->s2g0Y[left];

        // current implementation transfers from/to only immediate neighbours
        GMX_ASSERT(overlapSize <= myGridY && overlapSize <= rightGrid && overlapSize <= leftGrid,
                   "Exchange supported only with immediate neighbor");

        overlapY     = std::min(overlapSize, myGridY);
        overlapRight = std::min(overlapSize, rightGrid);
        overlapLeft  = std::min(overlapSize, leftGrid);

        // if only 2 PME ranks in Y-domain and overlap width more than slab width
        // just transfer all grid points from neighbor
        if (right == left && overlapRight + overlapLeft >= rightGrid)
        {
            overlapY     = myGridY;
            overlapRight = rightGrid;
            overlapLeft  = 0;
        }
    }

    for (int gridIndex = 0; gridIndex < pmeGpu->common->ngrids; gridIndex++)
    {
        MPI_Request req[16];
        int         count = 0;

        float* realGrid = pmeGpu->kernelParams->grid.d_realGrid[gridIndex];

        float* sendGridUp   = pmeGpu->archSpecific->d_sendGridUp;
        float* sendGridDown = pmeGpu->archSpecific->d_sendGridDown;
        float* recvGridUp   = pmeGpu->archSpecific->d_recvGridUp;
        float* recvGridDown = pmeGpu->archSpecific->d_recvGridDown;

        // no need to pack if slab-decomposition in X-dimension as data is already contiguous
        if (sizeY == 1)
        {
            int sendOffsetUp   = 0;
            int sendOffsetDown = (myGridX - overlapX) * localPmeSize[YY] * localPmeSize[ZZ];
            int recvOffsetUp = (localPmeSize[XX] - overlapUp) * localPmeSize[YY] * localPmeSize[ZZ];
            int recvOffsetDown = myGridX * localPmeSize[YY] * localPmeSize[ZZ];
            sendGridUp         = &realGrid[sendOffsetUp];
            sendGridDown       = &realGrid[sendOffsetDown];
            recvGridUp         = &realGrid[recvOffsetUp];
            recvGridDown       = &realGrid[recvOffsetDown];
        }
        else
        {
            // launch packing kernel
            packHaloDataInternal(pmeGpu,
                                 overlapX,
                                 overlapY,
                                 overlapUp,
                                 overlapLeft,
                                 myGridX,
                                 myGridY,
                                 localPmeSize,
                                 realGrid,
                                 sendGridUp,
                                 sendGridDown,
                                 pmeGpu->archSpecific->d_sendGridLeft,
                                 pmeGpu->archSpecific->d_sendGridRight,
                                 pmeGpu->archSpecific->d_sendGridUpLeft,
                                 pmeGpu->archSpecific->d_sendGridDownLeft,
                                 pmeGpu->archSpecific->d_sendGridUpRight,
                                 pmeGpu->archSpecific->d_sendGridDownRight);
        }

        // make sure data is ready on GPU before MPI communication
        pmeGpu->archSpecific->pmeStream_.synchronize();


        // major dimension
        if (sizeX > 1)
        {
            constexpr int mpiTag = 403; // Arbitrarily chosen

            // send data to up rank and recv from down rank
            MPI_Irecv(recvGridDown,
                      overlapDown * myGridY * localPmeSize[ZZ],
                      MPI_FLOAT,
                      down,
                      mpiTag,
                      pmeGpu->common->mpiCommX,
                      &req[count++]);

            MPI_Isend(sendGridUp,
                      overlapX * myGridY * localPmeSize[ZZ],
                      MPI_FLOAT,
                      up,
                      mpiTag,
                      pmeGpu->common->mpiCommX,
                      &req[count++]);


            if (overlapUp > 0)
            {
                // send data to down rank and recv from up rank
                MPI_Irecv(recvGridUp,
                          overlapUp * myGridY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          up,
                          mpiTag,
                          pmeGpu->common->mpiCommX,
                          &req[count++]);

                MPI_Isend(sendGridDown,
                          overlapX * myGridY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          down,
                          mpiTag,
                          pmeGpu->common->mpiCommX,
                          &req[count++]);
            }
        }

        // minor dimension
        if (sizeY > 1)
        {
            constexpr int mpiTag = 403; // Arbitrarily chosen

            // recv from right rank
            MPI_Irecv(pmeGpu->archSpecific->d_recvGridRight,
                      overlapRight * myGridX * localPmeSize[ZZ],
                      MPI_FLOAT,
                      right,
                      mpiTag,
                      pmeGpu->common->mpiCommY,
                      &req[count++]);

            // send data to left rank
            MPI_Isend(pmeGpu->archSpecific->d_sendGridLeft,
                      overlapY * myGridX * localPmeSize[ZZ],
                      MPI_FLOAT,
                      left,
                      mpiTag,
                      pmeGpu->common->mpiCommY,
                      &req[count++]);

            if (overlapLeft > 0)
            {
                // recv from left rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridLeft,
                          overlapLeft * myGridX * localPmeSize[ZZ],
                          MPI_FLOAT,
                          left,
                          mpiTag,
                          pmeGpu->common->mpiCommY,
                          &req[count++]);

                // send data to right rank
                MPI_Isend(pmeGpu->archSpecific->d_sendGridRight,
                          overlapY * myGridX * localPmeSize[ZZ],
                          MPI_FLOAT,
                          right,
                          mpiTag,
                          pmeGpu->common->mpiCommY,
                          &req[count++]);
            }
        }

        if (sizeX > 1 && sizeY > 1)
        {
            int rankUpLeft   = up * sizeY + left;
            int rankDownLeft = down * sizeY + left;

            int rankUpRight   = up * sizeY + right;
            int rankDownRight = down * sizeY + right;

            constexpr int mpiTag = 403; // Arbitrarily chosen

            // send data to up left and recv from down right rank
            MPI_Irecv(pmeGpu->archSpecific->d_recvGridDownRight,
                      overlapDown * overlapRight * localPmeSize[ZZ],
                      MPI_FLOAT,
                      rankDownRight,
                      mpiTag,
                      pmeGpu->common->mpiComm,
                      &req[count++]);

            MPI_Isend(pmeGpu->archSpecific->d_sendGridUpLeft,
                      overlapX * overlapY * localPmeSize[ZZ],
                      MPI_FLOAT,
                      rankUpLeft,
                      mpiTag,
                      pmeGpu->common->mpiComm,
                      &req[count++]);

            if (overlapLeft > 0)
            {
                // send data to up right rank and recv from down left rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridDownLeft,
                          overlapDown * overlapLeft * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankDownLeft,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);

                MPI_Isend(pmeGpu->archSpecific->d_sendGridUpRight,
                          overlapX * overlapY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankUpRight,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);
            }

            if (overlapUp > 0)
            {
                // send data to down left rank and recv from up right rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridUpRight,
                          overlapUp * overlapRight * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankUpRight,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);

                MPI_Isend(pmeGpu->archSpecific->d_sendGridDownLeft,
                          overlapX * overlapY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankDownLeft,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);
            }

            if (overlapUp > 0 && overlapLeft > 0)
            {
                // send data to down right rank and recv from up left rank
                MPI_Irecv(pmeGpu->archSpecific->d_recvGridUpLeft,
                          overlapUp * overlapLeft * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankUpLeft,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);

                MPI_Isend(pmeGpu->archSpecific->d_sendGridDownRight,
                          overlapX * overlapY * localPmeSize[ZZ],
                          MPI_FLOAT,
                          rankDownRight,
                          mpiTag,
                          pmeGpu->common->mpiComm,
                          &req[count++]);
            }
        }

        MPI_Waitall(count, req, MPI_STATUSES_IGNORE);

        // data is written at right place as part of MPI communication if slab-decomposition in X-dimension
        if (sizeY > 1)
        {
            // assign halo data
            assignHaloData(pmeGpu,
                           overlapUp,
                           overlapDown,
                           overlapLeft,
                           overlapRight,
                           myGridX,
                           myGridY,
                           localPmeSize,
                           realGrid,
                           recvGridUp,
                           recvGridDown,
                           pmeGpu->archSpecific->d_recvGridLeft,
                           pmeGpu->archSpecific->d_recvGridRight,
                           pmeGpu->archSpecific->d_recvGridUpLeft,
                           pmeGpu->archSpecific->d_recvGridDownLeft,
                           pmeGpu->archSpecific->d_recvGridUpRight,
                           pmeGpu->archSpecific->d_recvGridDownRight);
        }
    }
#else
    GMX_UNUSED_VALUE(pmeGpu);
#endif
}

template<bool pmeToFft>
void convertPmeGridToFftGrid(const PmeGpu*         pmeGpu,
                             float*                h_fftRealGrid,
                             gmx_parallel_3dfft_t* fftSetup,
                             const int             gridIndex)
{
    ivec localFftNData, localFftOffset, localFftSize;
    ivec localPmeSize;

    gmx_parallel_3dfft_real_limits(fftSetup[gridIndex], localFftNData, localFftOffset, localFftSize);

    localPmeSize[XX] = pmeGpu->kernelParams->grid.realGridSizePadded[XX];
    localPmeSize[YY] = pmeGpu->kernelParams->grid.realGridSizePadded[YY];
    localPmeSize[ZZ] = pmeGpu->kernelParams->grid.realGridSizePadded[ZZ];

    // this is true in case of slab decomposition
    if (localPmeSize[ZZ] == localFftSize[ZZ] && localPmeSize[YY] == localFftSize[YY])
    {
        int fftSize = localFftSize[ZZ] * localFftSize[YY] * localFftNData[XX];
        if (pmeToFft)
        {
            copyFromDeviceBuffer(h_fftRealGrid,
                                 &pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                 0,
                                 fftSize,
                                 pmeGpu->archSpecific->pmeStream_,
                                 pmeGpu->settings.transferKind,
                                 nullptr);
        }
        else
        {
            copyToDeviceBuffer(&pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                               h_fftRealGrid,
                               0,
                               fftSize,
                               pmeGpu->archSpecific->pmeStream_,
                               pmeGpu->settings.transferKind,
                               nullptr);
        }
    }
    else
    {
        // launch copy kernel
        // ToDo: Experiment with different block size and decide on optimal configuration

        // keeping same as warp size for better coalescing
        // Not keeping to higher value such as 64 to avoid high masked out
        // inactive threads as FFT grid sizes tend to be quite small
        const int threadsAlongZDim = 32;

        KernelLaunchConfig config;
        config.blockSize[0] = threadsAlongZDim;
        config.blockSize[1] = 4;
        config.blockSize[2] = 1;
        config.gridSize[0]  = (localFftNData[ZZ] + config.blockSize[0] - 1) / config.blockSize[0];
        config.gridSize[1]  = (localFftNData[YY] + config.blockSize[1] - 1) / config.blockSize[1];
        config.gridSize[2]  = localFftNData[XX];
        config.sharedMemorySize = 0;

        auto kernelFn = pmegrid_to_fftgrid<pmeToFft>;

        const auto kernelArgs =
                prepareGpuKernelArguments(kernelFn,
                                          config,
                                          &pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                          &h_fftRealGrid,
                                          &localFftNData,
                                          &localFftSize,
                                          &localPmeSize);

        launchGpuKernel(kernelFn,
                        config,
                        pmeGpu->archSpecific->pmeStream_,
                        nullptr,
                        "Convert PME grid to FFT grid",
                        kernelArgs);
    }

    if (pmeToFft)
    {
        pmeGpu->archSpecific->syncSpreadGridD2H.markEvent(pmeGpu->archSpecific->pmeStream_);
    }
}

template<bool pmeToFft>
void convertPmeGridToFftGrid(const PmeGpu* pmeGpu, DeviceBuffer<float>* d_fftRealGrid, const int gridIndex)
{
    ivec localPmeSize;

    ivec localFftNData, localFftSize;

    localPmeSize[XX] = pmeGpu->kernelParams->grid.realGridSizePadded[XX];
    localPmeSize[YY] = pmeGpu->kernelParams->grid.realGridSizePadded[YY];
    localPmeSize[ZZ] = pmeGpu->kernelParams->grid.realGridSizePadded[ZZ];

    localFftNData[XX] = pmeGpu->kernelParams->grid.localRealGridSize[XX];
    localFftNData[YY] = pmeGpu->kernelParams->grid.localRealGridSize[YY];
    localFftNData[ZZ] = pmeGpu->kernelParams->grid.localRealGridSize[ZZ];

    localFftSize[XX] = pmeGpu->kernelParams->grid.localRealGridSizePadded[XX];
    localFftSize[YY] = pmeGpu->kernelParams->grid.localRealGridSizePadded[YY];
    localFftSize[ZZ] = pmeGpu->kernelParams->grid.localRealGridSizePadded[ZZ];

    // this is true in case of slab decomposition
    if (localPmeSize[ZZ] == localFftSize[ZZ] && localPmeSize[YY] == localFftSize[YY])
    {
        int fftSize = localFftSize[ZZ] * localFftSize[YY] * localFftNData[XX];
        if (pmeToFft)
        {
            copyBetweenDeviceBuffers(d_fftRealGrid,
                                     &pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                     fftSize,
                                     pmeGpu->archSpecific->pmeStream_,
                                     pmeGpu->settings.transferKind,
                                     nullptr);
        }
        else
        {
            copyBetweenDeviceBuffers(&pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                     d_fftRealGrid,
                                     fftSize,
                                     pmeGpu->archSpecific->pmeStream_,
                                     pmeGpu->settings.transferKind,
                                     nullptr);
        }
    }
    else
    {
        // launch copy kernel
        // ToDo: Experiment with different block size and decide on optimal configuration

        // keeping same as warp size for better coalescing
        // Not keeping to higher value such as 64 to avoid high masked out
        // inactive threads as FFT grid sizes tend to be quite small
        const int threadsAlongZDim = 32;

        KernelLaunchConfig config;
        config.blockSize[0] = threadsAlongZDim;
        config.blockSize[1] = 4;
        config.blockSize[2] = 1;
        config.gridSize[0]  = (localFftNData[ZZ] + config.blockSize[0] - 1) / config.blockSize[0];
        config.gridSize[1]  = (localFftNData[YY] + config.blockSize[1] - 1) / config.blockSize[1];
        config.gridSize[2]  = localFftNData[XX];
        config.sharedMemorySize = 0;

        auto kernelFn = pmegrid_to_fftgrid<pmeToFft>;

        const auto kernelArgs =
                prepareGpuKernelArguments(kernelFn,
                                          config,
                                          &pmeGpu->kernelParams->grid.d_realGrid[gridIndex],
                                          d_fftRealGrid,
                                          &localFftNData,
                                          &localFftSize,
                                          &localPmeSize);

        launchGpuKernel(kernelFn,
                        config,
                        pmeGpu->archSpecific->pmeStream_,
                        nullptr,
                        "Convert PME grid to FFT grid",
                        kernelArgs);
    }
}

template void convertPmeGridToFftGrid<true>(const PmeGpu*         pmeGpu,
                                            float*                h_fftRealGrid,
                                            gmx_parallel_3dfft_t* fftSetup,
                                            const int             gridIndex);

template void convertPmeGridToFftGrid<false>(const PmeGpu*         pmeGpu,
                                             float*                h_fftRealGrid,
                                             gmx_parallel_3dfft_t* fftSetup,
                                             const int             gridIndex);

template void convertPmeGridToFftGrid<true>(const PmeGpu*        pmeGpu,
                                            DeviceBuffer<float>* d_fftRealGrid,
                                            const int            gridIndex);

template void convertPmeGridToFftGrid<false>(const PmeGpu*        pmeGpu,
                                             DeviceBuffer<float>* d_fftRealGrid,
                                             const int            gridIndex);
