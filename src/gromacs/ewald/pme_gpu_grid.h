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
 * \brief Implements PME halo exchange and PME-FFT grid conversion functions.
 *
 * \author Gaurav Garg <gaugarg@nvidia.com>
 *
 * \ingroup module_ewald
 */
#ifndef GMX_EWALD_PME_GPU_GRID_H
#define GMX_EWALD_PME_GPU_GRID_H

#include "gromacs/gpu_utils/devicebuffer_datatype.h"

struct PmeGpu;
typedef struct gmx_parallel_3dfft* gmx_parallel_3dfft_t;

/*! \libinternal \brief
 * Grid Halo exchange after PME spread
 * ToDo: Current implementation transfers halo region from/to only immediate neighbours
 * And, expects that overlapSize <= local grid width.
 * Implement exchange with multiple neighbors to remove this limitation
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 */
void pmeGpuGridHaloExchange(const PmeGpu* pmeGpu);

/*! \libinternal \brief
 * Grid reverse Halo exchange before PME gather
 * ToDo: Current implementation transfers halo region from/to only immediate neighbours
 * And, expects that overlapSize <= local grid width.
 * Implement exchange with multiple neighbors to remove this limitation
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 */
void pmeGpuGridHaloExchangeReverse(const PmeGpu* pmeGpu);

/*! \libinternal \brief
 * Copy PME Grid with overlap region to host FFT grid and vice-versa. Used in mixed mode PME decomposition
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 * \param[in]  h_fftRealGrid          FFT grid on host
 * \param[in]  fftSetup               Host FFT setup structure
 * \param[in]  gridIndex              Grid index which is to be converted
 *
 * \tparam  pmeToFft                  A boolean which tells if this is conversion from PME grid to FFT grid or reverse
 */
template<bool pmetofft>
void convertPmeGridToFftGrid(const PmeGpu*         pmeGpu,
                             float*                h_fftRealGrid,
                             gmx_parallel_3dfft_t* fftSetup,
                             int                   gridIndex);

/*! \libinternal \brief
 * Copy PME Grid with overlap region to device FFT grid and vice-versa. Used in full GPU PME decomposition
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 * \param[in]  d_fftRealGrid          FFT grid on device
 * \param[in]  gridIndex              Grid index which is to be converted
 *
 * \tparam  pmeToFft                  A boolean which tells if this is conversion from PME grid to FFT grid or reverse
 */
template<bool pmetofft>
void convertPmeGridToFftGrid(const PmeGpu* pmeGpu, DeviceBuffer<float>* d_fftRealGrid, int gridIndex);

extern template void convertPmeGridToFftGrid<true>(const PmeGpu* /*pmeGpu*/,
                                                   float* /*h_fftRealGrid*/,
                                                   gmx_parallel_3dfft_t* /*fftSetup*/,
                                                   int /*gridIndex*/);

extern template void convertPmeGridToFftGrid<false>(const PmeGpu* /*pmeGpu*/,
                                                    float* /*h_fftRealGrid*/,
                                                    gmx_parallel_3dfft_t* /*fftSetup*/,
                                                    int /*gridIndex*/);

extern template void convertPmeGridToFftGrid<true>(const PmeGpu* /*pmeGpu*/,
                                                   DeviceBuffer<float>* /*d_fftRealGrid*/,
                                                   int /*gridIndex*/);

extern template void convertPmeGridToFftGrid<false>(const PmeGpu* /*pmeGpu*/,
                                                    DeviceBuffer<float>* /*d_fftRealGrid*/,
                                                    int /*gridIndex*/);

#endif
