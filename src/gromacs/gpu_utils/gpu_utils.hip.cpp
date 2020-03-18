#include "hip/hip_runtime.h"
/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2010-2018, The GROMACS development team.
 * Copyright (c) 2019, by the GROMACS development team, led by
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
/*! \file
 *  \brief Define functions for detection and initialization for CUDA devices.
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */

#include "gmxpre.h"

#include "gpu_utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "gromacs/gpu_utils/cudautils.hip.h"
#include "gromacs/gpu_utils/pmalloc_cuda.h"
#include "gromacs/hardware/gpu_hw_info.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/snprintf.h"
#include "gromacs/utility/stringutil.h"

/*! \internal \brief
 * Max number of devices supported by ROCM (for consistency checking).
 */
static int rocm_max_device_count = 32;

/** Dummy kernel used for sanity checking. */
static __global__ void k_dummy_test(void) {}

static void checkCompiledTargetCompatibility(int deviceId, const hipDeviceProp_t& deviceProp)
{
    hipFuncAttributes attributes;
    hipError_t        stat = hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(k_dummy_test));

    if (hipErrorInvalidDeviceFunction == stat)
    {
        gmx_fatal(FARGS,
                  "The %s binary does not include support for the ROCM architecture of a "
                  "detected GPU: %s, ID #%d (compute capability %d.%d). "
                  "By default, GROMACS supports all architectures of compute "
                  "capability >= 3.0, so your GPU "
                  "might be rare, or some architectures were disabled in the build. "
                  "To work around this error, use the HIP_VISIBLE_DEVICES environment"
                  "variable to pass a list of GPUs that excludes the ID %d.",
                  gmx::getProgramContext().displayName(), deviceProp.name, deviceId,
                  deviceProp.major, deviceProp.minor, deviceId);
    }

    CU_RET_ERR(stat, "hipFuncGetAttributes failed");
}

bool isHostMemoryPinned(const void* h_ptr)
{
    hipPointerAttribute_t memoryAttributes;
    hipError_t           stat = hipPointerGetAttributes(&memoryAttributes, h_ptr);

    bool result = false;
    switch (stat)
    {
        case hipSuccess: result = true; break;

        case hipErrorInvalidValue:
            // If the buffer was not pinned, then it will not be recognized by ROCM at all
            result = false;
            // Reset the last error status
            hipGetLastError();
            break;

        default: CU_RET_ERR(stat, "Unexpected ROCM error");
    }
    return result;
}

/*!
 * \brief Runs GPU sanity checks.
 *
 * Runs a series of checks to determine that the given GPU and underlying ROCM
 * driver/runtime functions properly.
 *
 * \param[in]  dev_id      the device ID of the GPU or -1 if the device has already been initialized
 * \param[in]  dev_prop    The device properties structure
 * \returns                0 if the device looks OK
 *
 * TODO: introduce errors codes and handle errors more smoothly.
 */
static int do_sanity_checks(int dev_id, const hipDeviceProp_t& dev_prop)
{
    hipError_t cu_err;
    int         dev_count, id;

    cu_err = hipGetDeviceCount(&dev_count);
    if (cu_err != hipSuccess)
    {
        fprintf(stderr, "Error %d while querying device count: %s\n", cu_err, hipGetErrorString(cu_err));
        return -1;
    }

    /* no ROCM compatible device at all */
    if (dev_count == 0)
    {
        return -1;
    }

    /* things might go horribly wrong if cudart is not compatible with the driver */
    if (dev_count < 0 || dev_count > rocm_max_device_count)
    {
        return -1;
    }

    if (dev_id == -1) /* device already selected let's not destroy the context */
    {
        cu_err = hipGetDevice(&id);
        if (cu_err != hipSuccess)
        {
            fprintf(stderr, "Error %d while querying device id: %s\n", cu_err, hipGetErrorString(cu_err));
            return -1;
        }
    }
    else
    {
        id = dev_id;
        if (id > dev_count - 1) /* pfff there's no such device */
        {
            fprintf(stderr,
                    "The requested device with id %d does not seem to exist (device count=%d)\n",
                    dev_id, dev_count);
            return -1;
        }
    }

    /* both major & minor is 9999 if no ROCM capable devices are present */
    if (dev_prop.major == 9999 && dev_prop.minor == 9999)
    {
        return -1;
    }
    /* we don't care about emulation mode */
    if (dev_prop.major == 0)
    {
        return -1;
    }

    if (id != -1)
    {
        cu_err = hipSetDevice(id);
        if (cu_err != hipSuccess)
        {
            fprintf(stderr, "Error %d while switching to device #%d: %s\n", cu_err, id,
                    hipGetErrorString(cu_err));
            return -1;
        }
    }

    /* try to execute a dummy kernel */
    checkCompiledTargetCompatibility(dev_id, dev_prop);

    KernelLaunchConfig config;
    config.blockSize[0]       = 512;
    //const auto dummyArguments = prepareGpuKernelArguments(k_dummy_test, config);
    //launchGpuKernel(k_dummy_test, config, nullptr, "Dummy kernel", dummyArguments);
    launchGpuKernel(k_dummy_test, config, nullptr, "Dummy kernel");
    if (hipDeviceSynchronize() != hipSuccess)
    {
        return -1;
    }

    /* destroy context if we created one */
    if (id != -1)
    {
        cu_err = hipDeviceReset();
        CU_RET_ERR(cu_err, "hipDeviceReset failed");
    }

    return 0;
}

void init_gpu(const gmx_device_info_t* deviceInfo)
{
    hipError_t stat;

    assert(deviceInfo);

    stat = hipSetDevice(deviceInfo->id);
    if (stat != hipSuccess)
    {
        auto message = gmx::formatString("Failed to initialize GPU #%d", deviceInfo->id);
        CU_RET_ERR(stat, message.c_str());
    }

    if (debug)
    {
        fprintf(stderr, "Initialized GPU ID #%d: %s\n", deviceInfo->id, deviceInfo->prop.name);
    }
}

void free_gpu(const gmx_device_info_t* deviceInfo)
{
    // One should only attempt to clear the device context when
    // it has been used, but currently the only way to know that a GPU
    // device was used is that deviceInfo will be non-null.
    if (deviceInfo == nullptr)
    {
        return;
    }

    hipError_t stat;

    if (debug)
    {
        int gpuid;
        stat = hipGetDevice(&gpuid);
        CU_RET_ERR(stat, "hipGetDevice failed");
        fprintf(stderr, "Cleaning up context on GPU ID #%d\n", gpuid);
    }

    stat = hipDeviceReset();
    if (stat != hipSuccess)
    {
        gmx_warning("Failed to free GPU #%d: %s", deviceInfo->id, hipGetErrorString(stat));
    }
}

gmx_device_info_t* getDeviceInfo(const gmx_gpu_info_t& gpu_info, int deviceId)
{
    if (deviceId < 0 || deviceId >= gpu_info.n_dev)
    {
        gmx_incons("Invalid GPU deviceId requested");
    }
    return &gpu_info.gpu_dev[deviceId];
}

/*! \brief Returns true if the gpu characterized by the device properties is
 *  supported by the native gpu acceleration.
 *
 * \param[in] dev_prop  the CUDA device properties of the gpus to test.
 * \returns             true if the GPU properties passed indicate a compatible
 *                      GPU, otherwise false.
 */
static bool is_gmx_supported_gpu(const hipDeviceProp_t& dev_prop)
{
    return (dev_prop.major >= 3);
}

/*! \brief Checks if a GPU with a given ID is supported by the native GROMACS acceleration.
 *
 *  Returns a status value which indicates compatibility or one of the following
 *  errors: incompatibility or insanity (=unexpected behavior).
 *
 *  As the error handling only permits returning the state of the GPU, this function
 *  does not clear the CUDA runtime API status allowing the caller to inspect the error
 *  upon return. Note that this also means it is the caller's responsibility to
 *  reset the CUDA runtime state.
 *
 *  \param[in]  deviceId   the ID of the GPU to check.
 *  \param[in]  deviceProp the CUDA device properties of the device checked.
 *  \returns               the status of the requested device
 */
static int is_gmx_supported_gpu_id(int deviceId, const hipDeviceProp_t& deviceProp)
{
    if (!is_gmx_supported_gpu(deviceProp))
    {
        return egpuIncompatible;
    }

    /* TODO: currently we do not make a distinction between the type of errors
     * that can appear during sanity checks. This needs to be improved, e.g if
     * the dummy test kernel fails to execute with a "device busy message" we
     * should appropriately report that the device is busy instead of insane.
     */
    if (do_sanity_checks(deviceId, deviceProp) != 0)
    {
        return egpuInsane;
    }

    return egpuCompatible;
}

bool isGpuDetectionFunctional(std::string* errorMessage)
{
    hipError_t stat;
    int         driverVersion = -1;
    stat                      = hipDriverGetVersion(&driverVersion);
    GMX_ASSERT(stat != hipErrorInvalidValue,
               "An impossible null pointer was passed to hipDriverGetVersion");
    GMX_RELEASE_ASSERT(
            stat == hipSuccess,
            gmx::formatString("An unexpected value was returned from hipDriverGetVersion %s: %s",
                              hipGetErrorName(stat), hipGetErrorString(stat))
                    .c_str());
    bool foundDriver = (driverVersion > 0);
    if (!foundDriver)
    {
        // Can't detect GPUs if there is no driver
        if (errorMessage != nullptr)
        {
            errorMessage->assign("No valid ROCM driver found");
        }
        return false;
    }

    int numDevices;
    stat = hipGetDeviceCount(&numDevices);
    if (stat != hipSuccess)
    {
        if (errorMessage != nullptr)
        {
            /* hipGetDeviceCount failed which means that there is
             * something wrong with the machine: driver-runtime
             * mismatch, all GPUs being busy in exclusive mode,
             * invalid HIP_VISIBLE_DEVICES, or some other condition
             * which should result in GROMACS issuing at least a
             * warning. */
            errorMessage->assign(hipGetErrorString(stat));
        }

        // Consume the error now that we have prepared to handle
        // it. This stops it reappearing next time we check for
        // errors. Note that if HIP_VISIBLE_DEVICES does not contain
        // valid devices, then hipGetLastError returns the
        // (undocumented) hipErrorNoDevice, but this should not be a
        // problem as there should be no future HIP API calls.
        // NVIDIA bug report #2038718 has been filed.
        hipGetLastError();
        // Can't detect GPUs
        return false;
    }

    // We don't actually use numDevices here, that's not the job of
    // this function.
    return true;
}

void findGpus(gmx_gpu_info_t* gpu_info)
{
    assert(gpu_info);

    gpu_info->n_dev_compatible = 0;

    int         ndev;
    hipError_t stat = hipGetDeviceCount(&ndev);
    if (stat != hipSuccess)
    {
        GMX_THROW(gmx::InternalError(
                "Invalid call of findGpus() when HIP API returned an error, perhaps "
                "canDetectGpus() was not called appropriately beforehand."));
    }

    // We expect to start device support/sanity checks with a clean runtime error state
    gmx::ensureNoPendingCudaError("");

    gmx_device_info_t* devs;
    snew(devs, ndev);
    for (int i = 0; i < ndev; i++)
    {
        hipDeviceProp_t prop;
        memset(&prop, 0, sizeof(hipDeviceProp_t));
        stat = hipGetDeviceProperties(&prop, i);
        int checkResult;
        if (stat != hipSuccess)
        {
            // Will handle the error reporting below
            checkResult = egpuInsane;
        }
        else
        {
            checkResult = is_gmx_supported_gpu_id(i, prop);
        }

        devs[i].id   = i;
        devs[i].prop = prop;
        devs[i].stat = checkResult;

        if (checkResult == egpuCompatible)
        {
            gpu_info->n_dev_compatible++;
        }
        else
        {
            // TODO:
            //  - we inspect the HIP API state to retrieve and record any
            //    errors that occurred during is_gmx_supported_gpu_id() here,
            //    but this would be more elegant done within is_gmx_supported_gpu_id()
            //    and only return a string with the error if one was encountered.
            //  - we'll be reporting without rank information which is not ideal.
            //  - we'll end up warning also in cases where users would already
            //    get an error before mdrun aborts.
            //
            // Here we also clear the HIP API error state so potential
            // errors during sanity checks don't propagate.
            if ((stat = hipGetLastError()) != hipSuccess)
            {
                gmx_warning("An error occurred while sanity checking device #%d; %s: %s",
                            devs[i].id, hipGetErrorName(stat), hipGetErrorString(stat));
            }
        }
    }

    stat = hipPeekAtLastError();
    GMX_RELEASE_ASSERT(stat == hipSuccess,
                       gmx::formatString("We promise to return with clean ROCM state, but "
                                         "non-success state encountered: %s: %s",
                                         hipGetErrorName(stat), hipGetErrorString(stat))
                               .c_str());

    gpu_info->n_dev   = ndev;
    gpu_info->gpu_dev = devs;
}

void get_gpu_device_info_string(char* s, const gmx_gpu_info_t& gpu_info, int index)
{
    assert(s);

    if (index < 0 && index >= gpu_info.n_dev)
    {
        return;
    }

    gmx_device_info_t* dinfo = &gpu_info.gpu_dev[index];

    bool bGpuExists = (dinfo->stat != egpuNonexistent && dinfo->stat != egpuInsane);

    if (!bGpuExists)
    {
        sprintf(s, "#%d: %s, stat: %s", dinfo->id, "N/A", gpu_detect_res_str[dinfo->stat]);
    }
    else
    {
        sprintf(s, "#%d: AMD %s, compute cap.: %d.%d, stat: %s", dinfo->id,
                dinfo->prop.name, dinfo->prop.major, dinfo->prop.minor, gpu_detect_res_str[dinfo->stat]);
    }
}

int get_current_cuda_gpu_device_id(void)
{
    int gpuid;
    CU_RET_ERR(hipGetDevice(&gpuid), "hipGetDevice failed");

    return gpuid;
}

size_t sizeof_gpu_dev_info(void)
{
    return sizeof(gmx_device_info_t);
}

void startGpuProfiler(void)
{
//todo
/*
    if (cudaProfilerRun)
    {
        hipError_t stat;
        stat = hipProfilerStart();
        CU_RET_ERR(stat, "hipProfilerStart failed");
    }
*/
}

void stopGpuProfiler(void)
{
    /* Stopping the nvidia here allows us to eliminate the subsequent
       API calls from the trace, e.g. uninitialization and cleanup. */
//todo
/*
    if (cudaProfilerRun)
    {
        hipError_t stat;
        stat = hipProfilerStop();
        CU_RET_ERR(stat, "hipProfilerStop failed");
    }
*/
}

void resetGpuProfiler(void)
{
     /*
     * TODO: add a stop (or replace it with reset) when this will work correctly in CUDA.
     * stopGpuProfiler();
     */
//todo
/*
    if (cudaProfilerRun)
    {
        startGpuProfiler();
    }
*/
}

int gpu_info_get_stat(const gmx_gpu_info_t& info, int index)
{
    return info.gpu_dev[index].stat;
}

/*! \brief Check status returned from peer access CUDA call, and error out or warn appropriately
 * \param[in] stat           CUDA call return status
 * \param[in] gpuA           ID for GPU initiating peer access call
 * \param[in] gpuB           ID for remote GPU
 * \param[in] mdlog          Logger object
 * \param[in] cudaCallName   name of CUDA peer access call
 */
static void peerAccessCheckStat(const hipError_t    stat,
                                const int            gpuA,
                                const int            gpuB,
                                const gmx::MDLogger& mdlog,
                                const char*          cudaCallName)
{
    if ((stat == hipErrorInvalidDevice) || (stat == hipErrorInvalidValue))
    {
        std::string errorString =
                gmx::formatString("%s from GPU %d to GPU %d failed", cudaCallName, gpuA, gpuB);
        CU_RET_ERR(stat, errorString.c_str());
    }
    if (stat != hipSuccess)
    {
        GMX_LOG(mdlog.warning)
                .asParagraph()
                .appendTextFormatted(
                        "GPU peer access not enabled between GPUs %d and %d due to unexpected "
                        "return value from %s: %s",
                        gpuA, gpuB, cudaCallName, hipGetErrorString(stat));
    }
}

void setupGpuDevicePeerAccess(const std::vector<int>& gpuIdsToUse, const gmx::MDLogger& mdlog)
{
    hipError_t stat;

    // take a note of currently-set GPU
    int currentGpu;
    stat = hipGetDevice(&currentGpu);
    CU_RET_ERR(stat, "hipGetDevice in setupGpuDevicePeerAccess failed");

    std::string message = gmx::formatString(
            "Note: Peer access enabled between the following GPU pairs in the node:\n ");
    bool peerAccessEnabled = false;

    for (unsigned int i = 0; i < gpuIdsToUse.size(); i++)
    {
        int gpuA = gpuIdsToUse[i];
        stat     = hipSetDevice(gpuA);
        if (stat != hipSuccess)
        {
            GMX_LOG(mdlog.warning)
                    .asParagraph()
                    .appendTextFormatted(
                            "GPU peer access not enabled due to unexpected return value from "
                            "hipSetDevice(%d): %s",
                            gpuA, hipGetErrorString(stat));
            return;
        }
        for (unsigned int j = 0; j < gpuIdsToUse.size(); j++)
        {
            if (j != i)
            {
                int gpuB          = gpuIdsToUse[j];
                int canAccessPeer = 0;
                stat              = hipDeviceCanAccessPeer(&canAccessPeer, gpuA, gpuB);
                peerAccessCheckStat(stat, gpuA, gpuB, mdlog, "hipDeviceCanAccessPeer");

                if (canAccessPeer)
                {
                    stat = hipDeviceEnablePeerAccess(gpuB, 0);
                    peerAccessCheckStat(stat, gpuA, gpuB, mdlog, "hipDeviceEnablePeerAccess");

                    message           = gmx::formatString("%s%d->%d ", message.c_str(), gpuA, gpuB);
                    peerAccessEnabled = true;
                }
            }
        }
    }

    // re-set GPU to that originally set
    stat = hipSetDevice(currentGpu);
    if (stat != hipSuccess)
    {
        CU_RET_ERR(stat, "hipSetDevice in setupGpuDevicePeerAccess failed");
        return;
    }

    if (peerAccessEnabled)
    {
        GMX_LOG(mdlog.info).asParagraph().appendTextFormatted("%s", message.c_str());
    }
}
