/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2014,2015,2016,2017,2018,2019, by the GROMACS development team, led by
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

#include "gmxpre.h"

#include "cudautils.hip.h"

#include <cassert>
#include <cstdlib>

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/utility/gmxassert.h"

/*** Generic CUDA data operation wrappers ***/

// TODO: template on transferKind to avoid runtime conditionals
int cu_copy_D2H(void* h_dest, void* d_src, size_t bytes, GpuApiCallBehavior transferKind, hipStream_t s = nullptr)
{
    hipError_t stat;

    if (h_dest == nullptr || d_src == nullptr || bytes == 0)
    {
        return -1;
    }

    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            GMX_ASSERT(isHostMemoryPinned(h_dest), "Destination buffer was not pinned for CUDA");
            stat = hipMemcpyAsync(h_dest, d_src, bytes, hipMemcpyDeviceToHost, s);
            CU_RET_ERR(stat, "DtoH hipMemcpyAsync failed");
            break;

        case GpuApiCallBehavior::Sync:
            stat = hipMemcpy(h_dest, d_src, bytes, hipMemcpyDeviceToHost);
            CU_RET_ERR(stat, "DtoH hipMemcpy failed");
            break;

        default: throw;
    }

    return 0;
}

int cu_copy_D2H_sync(void* h_dest, void* d_src, size_t bytes)
{
    return cu_copy_D2H(h_dest, d_src, bytes, GpuApiCallBehavior::Sync);
}

/*!
 *  The copy is launched in stream s or if not specified, in stream 0.
 */
int cu_copy_D2H_async(void* h_dest, void* d_src, size_t bytes, hipStream_t s = nullptr)
{
    return cu_copy_D2H(h_dest, d_src, bytes, GpuApiCallBehavior::Async, s);
}

// TODO: template on transferKind to avoid runtime conditionals
int cu_copy_H2D(void* d_dest, const void* h_src, size_t bytes, GpuApiCallBehavior transferKind, hipStream_t s = nullptr)
{
    hipError_t stat;

    if (d_dest == nullptr || h_src == nullptr || bytes == 0)
    {
        return -1;
    }

    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            GMX_ASSERT(isHostMemoryPinned(h_src), "Source buffer was not pinned for CUDA");
            stat = hipMemcpyAsync(d_dest, h_src, bytes, hipMemcpyHostToDevice, s);
            CU_RET_ERR(stat, "HtoD hipMemcpyAsync failed");
            break;

        case GpuApiCallBehavior::Sync:
            stat = hipMemcpy(d_dest, h_src, bytes, hipMemcpyHostToDevice);
            CU_RET_ERR(stat, "HtoD hipMemcpy failed");
            break;

        default: throw;
    }

    return 0;
}

int cu_copy_H2D_sync(void* d_dest, const void* h_src, size_t bytes)
{
    return cu_copy_H2D(d_dest, h_src, bytes, GpuApiCallBehavior::Sync);
}

/*!
 *  The copy is launched in stream s or if not specified, in stream 0.
 */
int cu_copy_H2D_async(void* d_dest, const void* h_src, size_t bytes, hipStream_t s = nullptr)
{
    return cu_copy_H2D(d_dest, h_src, bytes, GpuApiCallBehavior::Async, s);
}

/*! \brief Set up texture object for an array of type T.
 *
 * Set up texture object for an array of type T and bind it to the device memory
 * \p d_ptr points to.
 *
 * \tparam[in] T        Raw data type
 * \param[out] texObj   texture object to initialize
 * \param[in]  d_ptr    pointer to device global memory to bind \p texObj to
 * \param[in]  sizeInBytes  size of memory area to bind \p texObj to
 */
template<typename T>
static void setup1DTexture(hipTextureObject_t& texObj, void* d_ptr, size_t sizeInBytes)
{
    assert(!c_disableCudaTextures);

    hipError_t      stat;
    hipResourceDesc rd;
    hipTextureDesc  td;

    memset(&rd, 0, sizeof(rd));
    rd.resType                = hipResourceTypeLinear;
    rd.res.linear.devPtr      = d_ptr;
    rd.res.linear.desc        = hipCreateChannelDesc<T>();
    rd.res.linear.sizeInBytes = sizeInBytes;

    memset(&td, 0, sizeof(td));
    td.readMode = hipReadModeElementType;
    stat        = hipCreateTextureObject(&texObj, &rd, &td, nullptr);
    CU_RET_ERR(stat, "hipCreateTextureObject failed");
}

template<typename T>
void initParamLookupTable(T*& d_ptr, hipTextureObject_t& texObj, const T* h_ptr, int numElem)
{
    const size_t sizeInBytes = numElem * sizeof(*d_ptr);
    hipError_t  stat        = hipMalloc((void**)&d_ptr, sizeInBytes);
    CU_RET_ERR(stat, "hipMalloc failed in initParamLookupTable");
    cu_copy_H2D_sync(d_ptr, (void*)h_ptr, sizeInBytes);

    if (!c_disableCudaTextures)
    {
        setup1DTexture<T>(texObj, d_ptr, sizeInBytes);
    }
}

template<typename T>
void destroyParamLookupTable(T* d_ptr, hipTextureObject_t texObj)
{
    if (!c_disableCudaTextures)
    {
        CU_RET_ERR(hipDestroyTextureObject(texObj), "hipDestroyTextureObject on texObj failed");
    }
    CU_RET_ERR(hipFree(d_ptr), "hipFree failed");
}

/*! \brief Add explicit instantiations of init/destroyParamLookupTable() here as needed.
 * One should also verify that the result of hipCreateChannelDesc<T>() during texture setup
 * looks reasonable, when instantiating the templates for new types - just in case.
 */
template void initParamLookupTable<float>(float*&, hipTextureObject_t&, const float*, int);
template void destroyParamLookupTable<float>(float*, hipTextureObject_t);
template void initParamLookupTable<int>(int*&, hipTextureObject_t&, const int*, int);
template void destroyParamLookupTable<int>(int*, hipTextureObject_t);
