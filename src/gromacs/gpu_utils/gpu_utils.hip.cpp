/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2010,2011,2012,2013,2014,2015,2016, The GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 *  \brief Define functions for detection and initialization for HIP devices.
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */

#include "gmxpre.h"

#include "gpu_utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_profile.h>
#ifdef GMX_USE_ROCTX
#include <roctx.h>
#endif

#include "gromacs/gpu_utils/hiputils.hpp"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool hipProfilerRun = ((getenv("NVPROF_ID") != nullptr));

bool isHostMemoryPinned(const void* h_ptr)
{
    hipPointerAttribute_t memoryAttributes;
    hipError_t           stat = hipPointerGetAttributes(&memoryAttributes, h_ptr);

    bool isPinned = false;
    switch (stat)
    {
        case hipSuccess:
            // In HIP 11.0, the field called memoryType in
            // hipPointerAttribute_t was replaced by a field called
            // type, along with a documented change of behavior when the
            // pointer passed to hipPointerGetAttributes is to
            // non-registered host memory. That change means that this
            // code needs conditional compilation and different
            // execution paths to function with all supported versions.
//#if HIPRT_VERSION < 11 * 1000
//            isPinned = true;
//#else
            isPinned = (memoryAttributes.memoryType == hipMemoryTypeHost);
//#endif
            break;

        case hipErrorInvalidValue:
            // If the buffer was not pinned, then it will not be recognized by HIP at all
            isPinned = false;
            // Reset the last error status
            hipGetLastError();
            break;

        default: HIP_RET_ERR(stat, "Unexpected HIP error");
    }
    return isPinned;
}

void startGpuProfiler()
{
    /* The NVPROF_ID environment variable is set by nvprof and indicates that
       mdrun is executed in the HIP profiler.
       If nvprof was run is with "--profile-from-start off", the profiler will
       be started here. This way we can avoid tracing the HIP events from the
       first part of the run. Starting the profiler again does nothing.
     */
    if (hipProfilerRun)
    {
        hipError_t stat;
        stat = hipProfilerStart();
        HIP_RET_ERR(stat, "hipProfilerStart failed");
    }
}

void stopGpuProfiler()
{
    /* Stopping the nvidia here allows us to eliminate the subsequent
       API calls from the trace, e.g. uninitialization and cleanup. */
    if (hipProfilerRun)
    {
        hipError_t stat;
        stat = hipProfilerStop();
        HIP_RET_ERR(stat, "hipProfilerStop failed");
    }
}

void resetGpuProfiler()
{
    /* With HIP <=7.5 the profiler can't be properly reset; we can only start
     *  the profiling here (can't stop it) which will achieve the desired effect if
     *  the run was started with the profiling disabled.
     *
     * TODO: add a stop (or replace it with reset) when this will work correctly in HIP.
     * stopGpuProfiler();
     */
    if (hipProfilerRun)
    {
        startGpuProfiler();
    }
}

/*! \brief Check and act on status returned from peer access HIP call
 *
 * If status is "hipSuccess", we continue. If
 * "hipErrorPeerAccessAlreadyEnabled", then peer access has already
 * been enabled so we ignore. If "hipErrorInvalidDevice" then the
 * run is trying to access an invalid GPU, so we throw an error. If
 * "hipErrorInvalidValue" then there is a problem with the arguments
 * to the HIP call, and we throw an error. These cover all expected
 * statuses, but if any other is returned we issue a warning and
 * continue.
 *
 * \param[in] stat           HIP call return status
 * \param[in] gpuA           ID for GPU initiating peer access call
 * \param[in] gpuB           ID for remote GPU
 * \param[in] mdlog          Logger object
 * \param[in] hipCallName    name of HIP peer access call
 */
static void peerAccessCheckStat(const hipError_t    stat,
                                const int            gpuA,
                                const int            gpuB,
                                const gmx::MDLogger& mdlog,
                                const char*          hipCallName)
{

    if (stat == hipErrorPeerAccessAlreadyEnabled)
    {
        // Since peer access has already been enabled, this error can safely be ignored.
        // Now clear the error internally within HIP:
        hipGetLastError();
        return;
    }
    if ((stat == hipErrorInvalidDevice) || (stat == hipErrorInvalidValue))
    {
        std::string errorString =
                gmx::formatString("%s from GPU %d to GPU %d failed", hipCallName, gpuA, gpuB);
        HIP_RET_ERR(stat, errorString);
    }
    if (stat != hipSuccess)
    {
        GMX_LOG(mdlog.warning)
                .asParagraph()
                .appendTextFormatted(
                        "GPU peer access not enabled between GPUs %d and %d due to unexpected "
                        "return value from %s. %s",
                        gpuA,
                        gpuB,
                        hipCallName,
                        gmx::getDeviceErrorString(stat).c_str());
        // Clear the error internally within HIP
        hipGetLastError();
    }
}

void setupGpuDevicePeerAccess(const std::vector<int>& gpuIdsToUse, const gmx::MDLogger& mdlog)
{
    hipError_t stat;

    // take a note of currently-set GPU
    int currentGpu;
    stat = hipGetDevice(&currentGpu);
    HIP_RET_ERR(stat, "hipGetDevice in setupGpuDevicePeerAccess failed");

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
                            "hipSetDevice(%d). %s",
                            gpuA,
                            gmx::getDeviceErrorString(stat).c_str());
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
        HIP_RET_ERR(stat, "hipSetDevice in setupGpuDevicePeerAccess failed");
        return;
    }

    if (peerAccessEnabled)
    {
        GMX_LOG(mdlog.info).asParagraph().appendTextFormatted("%s", message.c_str());
    }
}

void hipRangePush(const char* msg){
#ifdef GMX_USE_ROCTX
    roctxRangePush(msg);
#endif
}

void hipRangePop(){
#ifdef GMX_USE_ROCTX
    roctxRangePop();
#endif
}
