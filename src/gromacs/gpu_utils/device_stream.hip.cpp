/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2020- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
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
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief Implements the DeviceStream for CUDA.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_gpu_utils
 */
#include "gmxpre.h"

#include "device_stream.h"

#include "gromacs/gpu_utils/hiputils.hpp"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include <iostream>

DeviceStream::DeviceStream(const DeviceContext& /* deviceContext */,
                           DeviceStreamPriority priority,
                           const bool /* useTiming */)
{
    std::cout << "DeviceStream() constructor" << std::endl;
    std::cout.flush();

    hipError_t stat;

    if (priority == DeviceStreamPriority::Normal)
    {
        stream_pointer_ = new hipStream_t;
        stat = hipStreamCreate(stream_pointer_);
        gmx::checkDeviceError(stat, "Could not create HIP stream.");
    }
    else if (priority == DeviceStreamPriority::High)
    {
        stream_pointer_ = new hipStream_t;
        // Note that the device we're running on does not have to
        // support priorities, because we are querying the priority
        // range, which in that case will be a single value.
        int highestPriority;
        stat = hipDeviceGetStreamPriorityRange(nullptr, &highestPriority);
        gmx::checkDeviceError(stat, "Could not query HIP stream priority range.");

        stat = hipStreamCreateWithPriority(stream_pointer_, hipStreamDefault, highestPriority);
        gmx::checkDeviceError(stat, "Could not create HIP stream with high priority.");
    }

    std::cout << "DeviceStream() constructor" << stream_pointer_ << " ; " << *stream_pointer_ <<  std::endl;
    std::cout.flush();
}

DeviceStream::~DeviceStream()
{
    if (isValid())
    {
        std::cout << "~DeviceStream() " << *stream_pointer_ << ", " << stream_pointer_ << std::endl;
        std::cout.flush();

        hipError_t stat = hipStreamDestroy(*stream_pointer_);
        GMX_RELEASE_ASSERT(stat == hipSuccess,
                           ("Failed to release HIP stream. " + gmx::getDeviceErrorString(stat)).c_str());

        delete stream_pointer_;
        stream_pointer_ = nullptr;

        std::cout << "~DeviceStream() " << stream_pointer_ << std::endl;
    }
}

hipStream_t DeviceStream::stream() const
{
    return *stream_pointer_;
}

hipStream_t* DeviceStream::stream_pointer() const
{
    return stream_pointer_;
}

bool DeviceStream::isValid() const
{
    std::cout << "isValid() " << *stream_pointer_ << ", " << stream_pointer_ << std::endl;
    std::cout.flush();

    return (stream_pointer_ != nullptr && *stream_pointer_ != nullptr);
}

void DeviceStream::synchronize() const
{
    hipError_t stat = hipStreamSynchronize(*stream_pointer_);
    GMX_RELEASE_ASSERT(stat == hipSuccess,
                       ("hipStreamSynchronize failed. " + gmx::getDeviceErrorString(stat)).c_str());
}
