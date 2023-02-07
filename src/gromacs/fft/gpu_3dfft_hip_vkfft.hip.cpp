/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2016- The GROMACS Authors
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
 *  \brief Implements GPU 3D FFT routines for HIP.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft_hip_vkfft.hpp"

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

Gpu3dFft::ImplHipVkFft::ImplHipVkFft(bool allocateRealGrid,
                                     MPI_Comm /*comm*/,
                                     ArrayRef<const int> gridSizesInXForEachRank,
                                     ArrayRef<const int> gridSizesInYForEachRank,
                                     const int /*nz*/,
                                     bool performOutOfPlaceFFT,
                                     const DeviceContext& context,
                                     const DeviceStream&  pmeStream,
                                     ivec                 realGridSize,
                                     ivec                 realGridSizePadded,
                                     ivec                 complexGridSizePadded,
                                     DeviceBuffer<float>* realGrid,
                                     DeviceBuffer<float>* complexGrid) :
    Gpu3dFft::Impl::Impl(performOutOfPlaceFFT), realGrid_(reinterpret_cast<float*>(*realGrid))
{
    GMX_RELEASE_ASSERT(allocateRealGrid == false, "Grids needs to be pre-allocated");
    GMX_RELEASE_ASSERT(gridSizesInXForEachRank.size() == 1 && gridSizesInYForEachRank.size() == 1,
                       "FFT decomposition not implemented with hipFFT backend");

   allocateComplexGrid(complexGridSizePadded, realGrid, complexGrid, context);

    const int complexGridSizePaddedTotal =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
    const int realGridSizePaddedTotal =
            realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

    GMX_RELEASE_ASSERT(realGrid_, "Bad (null) input real-space grid");
    GMX_RELEASE_ASSERT(complexGrid_, "Bad (null) input complex grid");

    configuration = {};
    appR2C = {};
    configuration.FFTdim = 3;
    configuration.size[0] = realGridSize[ZZ];
    configuration.size[1] = realGridSize[YY];
    configuration.size[2] = realGridSize[XX];

    configuration.performR2C = 1;
    //configuration.disableMergeSequencesR2C = 1;
    configuration.device = (hipDevice_t*)malloc(sizeof(hipDevice_t));
    hipError_t result = hipGetDevice(configuration.device);
    configuration.stream = pmeStream.stream_pointer();
    configuration.num_streams=1;

    uint64_t bufferSize = complexGridSizePadded[XX]* complexGridSizePadded[YY]* complexGridSizePadded[ZZ] * 2 * sizeof(float);
    configuration.bufferSize=&bufferSize;
    configuration.aimThreads = 64;
    configuration.bufferStride[0] = complexGridSizePadded[ZZ];
    configuration.bufferStride[1] = complexGridSizePadded[ZZ]* complexGridSizePadded[YY];
    configuration.bufferStride[2] = complexGridSizePadded[ZZ]* complexGridSizePadded[YY]* complexGridSizePadded[XX];
    configuration.buffer = (void**)&complexGrid_;

    configuration.isInputFormatted = 1;
    configuration.inverseReturnToInputBuffer = 1;
    uint64_t inputBufferSize = realGridSizePadded[XX]* realGridSizePadded[YY]* realGridSizePadded[ZZ] * sizeof(float);
    configuration.inputBufferSize = &inputBufferSize;
    configuration.inputBufferStride[0] = realGridSizePadded[ZZ];
    configuration.inputBufferStride[1] = realGridSizePadded[ZZ]* realGridSizePadded[YY];
    configuration.inputBufferStride[2] = realGridSizePadded[ZZ]* realGridSizePadded[YY]* realGridSizePadded[XX];
    configuration.inputBuffer = (void**)&realGrid_;
    VkFFTResult resFFT = initializeVkFFT(&appR2C, configuration);
    if (resFFT!=VKFFT_SUCCESS) printf ("VkFFT error: %d\n", resFFT);
}

Gpu3dFft::ImplHipVkFft::~ImplHipVkFft()
{
    deallocateComplexGrid();
    deleteVkFFT(&appR2C);
    free(configuration.device);
}

void Gpu3dFft::ImplHipVkFft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {
        resFFT = VkFFTAppend(&appR2C, -1, NULL);
        if (resFFT!=VKFFT_SUCCESS) printf ("VkFFT error: %d\n", resFFT);
    }
    else
    {
        resFFT = VkFFTAppend(&appR2C, 1, NULL);
        if (resFFT!=VKFFT_SUCCESS) printf ("VkFFT error: %d\n", resFFT);
    }
}

} // namespace gmx
