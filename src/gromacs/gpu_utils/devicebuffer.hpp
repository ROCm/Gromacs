/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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
#ifndef GMX_GPU_UTILS_DEVICEBUFFER_HPP
#define GMX_GPU_UTILS_DEVICEBUFFER_HPP

/*! \libinternal \file
 *  \brief Implements the DeviceBuffer type and routines for HIP.
 *  Should only be included directly by the main DeviceBuffer file devicebuffer.h.
 *  TODO: the intent is for DeviceBuffer to become a class.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *
 *  \inlibraryapi
 */

#include "gromacs/gpu_utils/hip_arch_utils.hpp"
#include "gromacs/gpu_utils/hiputils.hpp"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gpu_utils.h" //only for GpuApiCallBehavior
#include "gromacs/gpu_utils/gputraits.hpp"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

/*! \brief
 * Allocates a device-side buffer.
 * It is currently a caller's responsibility to call it only on not-yet allocated buffers.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer.
 * \param[in]     numValues            Number of values to accommodate.
 * \param[in]     deviceContext        The buffer's dummy device  context - not managed explicitly in HIP RT.
 */
template<typename ValueType>
void allocateDeviceBuffer(DeviceBuffer<ValueType>* buffer, size_t numValues, const DeviceContext& /* deviceContext */)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
#ifdef GMX_UNIFIED_MEM
    const int align = (numValues >= 256) ? 256 : __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    *buffer = new (std::align_val_t(align)) ValueType[numValues]; 
#else
    hipError_t stat = hipMalloc(buffer, numValues * sizeof(ValueType));
    GMX_RELEASE_ASSERT(
            stat == hipSuccess,
            ("Allocation of the device buffer failed. " + gmx::getDeviceErrorString(stat)).c_str());
#endif
}

/*! \brief
 * Frees a device-side buffer.
 * This does not reset separately stored size/capacity integers,
 * as this is planned to be a destructor of DeviceBuffer as a proper class,
 * and no calls on \p buffer should be made afterwards.
 *
 * \param[in] buffer  Pointer to the buffer to free.
 */
template<typename DeviceBuffer>
void freeDeviceBuffer(DeviceBuffer* buffer)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    if (*buffer)
    {
#ifdef GMX_UNIFIED_MEM
        delete *buffer;
#else
        hipError_t stat = hipFree(*buffer);
        GMX_RELEASE_ASSERT(
                stat == hipSuccess,
                ("Freeing of the device buffer failed. " + gmx::getDeviceErrorString(stat)).c_str());
#endif
    }
}

/*! \brief
 * Performs the host-to-device data copy, synchronous or asynchronously on request.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer
 * \param[in]     hostBuffer           Pointer to the raw host-side memory, also typed \p ValueType
 * \param[in]     startingOffset       Offset (in values) at the device-side buffer to copy into.
 * \param[in]     numValues            Number of values to copy.
 * \param[in]     deviceStream         GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind         Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent          A dummy pointer to the H2D copy timing event to be filled in.
 *                                     Not used in HIP implementation.
 */
template<typename ValueType>
void copyToDeviceBuffer(DeviceBuffer<ValueType>* buffer,
                        const ValueType*         hostBuffer,
                        size_t                   startingOffset,
                        size_t                   numValues,
                        const DeviceStream&      deviceStream,
                        GpuApiCallBehavior       transferKind,
                        CommandEvent* /*timingEvent*/)
{
    if (numValues == 0)
    {
        return;
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(hostBuffer, "needs a host buffer pointer");
    hipError_t  stat;
    const size_t bytes = numValues * sizeof(ValueType);
#ifndef GMX_UNIFIED_MEM
    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            GMX_ASSERT(isHostMemoryPinned(hostBuffer), "Source host buffer was not pinned for HIP");
            stat = hipMemcpyAsync(*reinterpret_cast<ValueType**>(buffer) + startingOffset,
                                   hostBuffer,
                                   bytes,
                                   hipMemcpyHostToDevice,
                                   deviceStream.stream());
            GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Asynchronous H2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
            break;

        case GpuApiCallBehavior::Sync:
            stat = hipMemcpy(*reinterpret_cast<ValueType**>(buffer) + startingOffset,
                              hostBuffer,
                              bytes,
                              hipMemcpyHostToDevice);
            GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Synchronous H2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
            break;

        default: throw;
    }
#else
    stat = hipStreamSynchronize(deviceStream.stream());
    memcpy(*reinterpret_cast<ValueType**>(buffer) + startingOffset, hostBuffer, bytes);
    GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Synchronous H2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
#endif
}

/*! \brief
 * Performs the device-to-host data copy, synchronous or asynchronously on request.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] hostBuffer           Pointer to the raw host-side memory, also typed \p ValueType
 * \param[in]     buffer               Pointer to the device-side buffer
 * \param[in]     startingOffset       Offset (in values) at the device-side buffer to copy from.
 * \param[in]     numValues            Number of values to copy.
 * \param[in]     deviceStream         GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind         Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent          A dummy pointer to the H2D copy timing event to be filled in.
 *                                     Not used in HIP implementation.
 */
template<typename ValueType>
void copyFromDeviceBuffer(ValueType*               hostBuffer,
                          DeviceBuffer<ValueType>* buffer,
                          size_t                   startingOffset,
                          size_t                   numValues,
                          const DeviceStream&      deviceStream,
                          GpuApiCallBehavior       transferKind,
                          CommandEvent* /*timingEvent*/)
{
    if (numValues == 0)
    {
        return;
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(hostBuffer, "needs a host buffer pointer");

    hipError_t  stat;
    const size_t bytes = numValues * sizeof(ValueType);
#ifndef GMX_UNIFIED_MEM
    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            GMX_ASSERT(isHostMemoryPinned(hostBuffer),
                       "Destination host buffer was not pinned for HIP");
            stat = hipMemcpyAsync(hostBuffer,
                                   *reinterpret_cast<ValueType**>(buffer) + startingOffset,
                                   bytes,
                                   hipMemcpyDeviceToHost,
                                   deviceStream.stream());
            GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Asynchronous D2H copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
            break;

        case GpuApiCallBehavior::Sync:
            stat = hipMemcpy(hostBuffer,
                              *reinterpret_cast<ValueType**>(buffer) + startingOffset,
                              bytes,
                              hipMemcpyDeviceToHost);
            GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Synchronous D2H copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
            break;

        default: throw;
    }
#else
    stat = hipStreamSynchronize(deviceStream.stream());
    memcpy(hostBuffer, *reinterpret_cast<ValueType**>(buffer) + startingOffset, bytes);
    GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Synchronous H2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
#endif
}

/*! \brief
 * Performs the device-to-device data copy, synchronous or asynchronously on request.
 *
 * \tparam        ValueType                Raw value type of the \p buffer.
 * \param[in,out] destinationDeviceBuffer  Device-side buffer to copy to
 * \param[in]     sourceDeviceBuffer       Device-side buffer to copy from
 * \param[in]     numValues                Number of values to copy.
 * \param[in]     deviceStream             GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind             Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent              A dummy pointer to the D2D copy timing event to be filled
 * in. Not used in HIP implementation.
 */
template<typename ValueType>
void copyBetweenDeviceBuffers(DeviceBuffer<ValueType>* destinationDeviceBuffer,
                              DeviceBuffer<ValueType>* sourceDeviceBuffer,
                              size_t                   numValues,
                              const DeviceStream&      deviceStream,
                              GpuApiCallBehavior       transferKind,
                              CommandEvent* /*timingEvent*/)
{
    if (numValues == 0)
    {
        return;
    }
    GMX_ASSERT(destinationDeviceBuffer, "needs a destination buffer pointer");
    GMX_ASSERT(sourceDeviceBuffer, "needs a source buffer pointer");

    hipError_t  stat;
    const size_t bytes = numValues * sizeof(ValueType);
#ifndef GMX_UNIFIED_MEM
    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            stat = hipMemcpyAsync(*destinationDeviceBuffer,
                                   *sourceDeviceBuffer,
                                   bytes,
                                   hipMemcpyDeviceToDevice,
                                   deviceStream.stream());
            GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Asynchronous D2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
            break;

        case GpuApiCallBehavior::Sync:
            stat = hipMemcpy(*destinationDeviceBuffer, *sourceDeviceBuffer, bytes, hipMemcpyDeviceToDevice);
            GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Synchronous D2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
            break;

        default: throw;
    }
#else
    // conventional memcpy. do we need the host involvement here?
    stat = hipStreamSynchronize(deviceStream.stream());
    memcpy(*sourceDeviceBuffer, *(destinationDeviceBuffer), bytes);
    GMX_RELEASE_ASSERT(
                    stat == hipSuccess,
                    ("Synchronous D2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());
#endif
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
__device__ inline
void block_store_direct_striped(unsigned int flat_id,
                                T* block_output,
                                T (&items)[ItemsPerThread],
                                unsigned int valid)
{
    T* thread_iter = block_output + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
             thread_iter[offset] = items[item];
        }
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
__device__ inline
void block_store_direct_striped(unsigned int flat_id,
                                T* block_output,
                                T (&items)[ItemsPerThread])
{
    T* thread_iter = block_output + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
         thread_iter[item * BlockSize] = items[item];
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
__launch_bounds__(BlockSize)
__global__ void kernel_fill(
    T* dst_ptr,
    T value,
    size_t size)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = threadIdx.x;
    const unsigned int flat_block_id = blockIdx.x;
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = (size + items_per_block - 1)/items_per_block;
    const auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);

    T values[ItemsPerThread];

    #pragma unroll
    for(unsigned int index = 0; index < ItemsPerThread; index++ )
    {
        values[index] = value;
    }

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_store_direct_striped<BlockSize, ItemsPerThread>(
            flat_id,
            dst_ptr + block_offset,
            values,
            valid_in_last_block
        );
    }
    else
    {
        block_store_direct_striped<BlockSize, ItemsPerThread>(
            flat_id,
            dst_ptr + block_offset,
            values
        );
    }
}

/*! \brief
 * Clears the device buffer asynchronously.
 *
 * \tparam        ValueType       Raw value type of the \p buffer.
 * \param[in,out] buffer          Pointer to the device-side buffer
 * \param[in]     startingOffset  Offset (in values) at the device-side buffer to start clearing at.
 * \param[in]     numValues       Number of values to clear.
 * \param[in]     deviceStream    GPU stream.
 */
template<typename ValueType>
void clearDeviceBufferAsync(DeviceBuffer<ValueType>* buffer,
                            size_t                   startingOffset,
                            size_t                   numValues,
                            const DeviceStream&      deviceStream)
{
    if (numValues == 0)
    {
        return;
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");
    const size_t bytes   = numValues * sizeof(ValueType);
    const char   pattern = 0;

#ifdef GMX_UNIFIED_MEM
    hipError_t stat = hipStreamSynchronize(deviceStream.stream());
    memset(*reinterpret_cast<ValueType**>(buffer) + startingOffset, pattern, bytes);
#else
    hipError_t stat = hipMemsetAsync(
            *reinterpret_cast<ValueType**>(buffer) + startingOffset, pattern, bytes, deviceStream.stream());
#endif

    /*KernelLaunchConfig config;
    constexpr unsigned int blockSize = 256;
    constexpr unsigned int itemsPerThread = 12;
    constexpr unsigned int itemsPerBlock = blockSize * itemsPerThread;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel_fill<blockSize,itemsPerThread, ValueType>),
        dim3((numValues + itemsPerBlock - 1) / itemsPerBlock, 1, 1),
        dim3(blockSize, 1, 1),
        0,
        deviceStream.stream(),
        *reinterpret_cast<ValueType**>(buffer) + startingOffset,
        ValueType(0),
        numValues
    );*/

    GMX_RELEASE_ASSERT(stat == hipSuccess,
                       ("Couldn't clear the device buffer. " + gmx::getDeviceErrorString(stat)).c_str());
}

/*! \brief Check the validity of the device buffer.
 *
 * Checks if the buffer is not nullptr.
 *
 * \todo Add checks on the buffer size when it will be possible.
 *
 * \param[in] buffer        Device buffer to be checked.
 * \param[in] requiredSize  Number of elements that the buffer will have to accommodate.
 *
 * \returns Whether the device buffer can be set.
 */
template<typename T>
gmx_unused static bool checkDeviceBuffer(DeviceBuffer<T> buffer, gmx_unused int requiredSize)
{
    GMX_ASSERT(buffer != nullptr, "The device pointer is nullptr");
    return buffer != nullptr;
}

//! Device texture wrapper.
using DeviceTexture = hipTextureObject_t;

/*! \brief Create a texture object for an array of type ValueType.
 *
 * Creates the device buffer, copies data and binds texture object for an array of type ValueType.
 *
 * \todo Test if using textures is still relevant on modern hardware.
 *
 * \tparam      ValueType      Raw data type.
 *
 * \param[out]  deviceBuffer   Device buffer to store data in.
 * \param[out]  deviceTexture  Device texture object to initialize.
 * \param[in]   hostBuffer     Host buffer to get date from
 * \param[in]   numValues      Number of elements in the buffer.
 * \param[in]   deviceContext  GPU device context.
 */
template<typename ValueType>
void initParamLookupTable(DeviceBuffer<ValueType>* deviceBuffer,
                          DeviceTexture*           deviceTexture,
                          const ValueType*         hostBuffer,
                          int                      numValues,
                          const DeviceContext&     deviceContext)
{
    if (numValues == 0)
    {
        return;
    }
    GMX_ASSERT(hostBuffer, "Host buffer should be specified.");

    allocateDeviceBuffer(deviceBuffer, numValues, deviceContext);

    const size_t sizeInBytes = numValues * sizeof(ValueType);

#ifdef GMX_UNIFIED_MEM
    // no stream is passed to this functions so I assume that this is not performance critical
    hipError_t stat = hipDeviceSynchronize();
    memcpy(*reinterpret_cast<ValueType**>(deviceBuffer), hostBuffer, sizeInBytes);
#else
    hipError_t stat = hipMemcpy(
            *reinterpret_cast<ValueType**>(deviceBuffer), hostBuffer, sizeInBytes, hipMemcpyHostToDevice);
#endif

    GMX_RELEASE_ASSERT(stat == hipSuccess,
                       ("Synchronous H2D copy failed. " + gmx::getDeviceErrorString(stat)).c_str());

    if (!c_disableHipTextures)
    {
        hipResourceDesc rd;
        hipTextureDesc  td;

        memset(&rd, 0, sizeof(rd));
        rd.resType                = hipResourceTypeLinear;
        rd.res.linear.devPtr      = *deviceBuffer;
        rd.res.linear.desc        = hipCreateChannelDesc<ValueType>();
        rd.res.linear.sizeInBytes = sizeInBytes;

        memset(&td, 0, sizeof(td));
        td.readMode = hipReadModeElementType;
        stat        = hipCreateTextureObject(deviceTexture, &rd, &td, nullptr);
        GMX_RELEASE_ASSERT(
                stat == hipSuccess,
                ("Binding of the texture object failed. " + gmx::getDeviceErrorString(stat)).c_str());
    }
}

/*! \brief Unbind the texture and release the HIP texture object.
 *
 * \tparam         ValueType      Raw data type
 *
 * \param[in,out]  deviceBuffer   Device buffer to store data in.
 * \param[in,out]  deviceTexture  Device texture object to unbind.
 */
template<typename ValueType>
void destroyParamLookupTable(DeviceBuffer<ValueType>* deviceBuffer, const DeviceTexture* deviceTexture)
{
    if (!c_disableHipTextures && deviceTexture && deviceBuffer)
    {
        hipError_t stat = hipDestroyTextureObject(*deviceTexture);
        GMX_RELEASE_ASSERT(
                stat == hipSuccess,
                ("Destruction of the texture object failed. " + gmx::getDeviceErrorString(stat)).c_str());
    }
    freeDeviceBuffer(deviceBuffer);
}

#endif
