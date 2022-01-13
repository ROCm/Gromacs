/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
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
 *  \brief Define CUDA implementation of nbnxn_gpu.h
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */
#include "gmxpre.h"

#include "config.h"

#include <assert.h>
#include <stdlib.h>

#include "gromacs/nbnxm/nbnxm_gpu.h"

#if defined(_MSVC)
#    include <limits>
#endif


#include "nbnxm_cuda.h"

#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gpueventsynchronizer_hip.h"
#include "gromacs/gpu_utils/typecasts_hip.h"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/hardware/device_information.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_common.h"
#include "gromacs/nbnxm/gpu_common_utils.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/grid.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/gmxassert.h"

#include "nbnxm_buffer_ops_kernels_hip.h"
#include "nbnxm_cuda_types.h"

/***** The kernel declarations/definitions come here *****/

/* Top-level kernel declaration generation: will generate through multiple
 * inclusion the following flavors for all kernel declarations:
 * - force-only output;
 * - force and energy output;
 * - force-only with pair list pruning;
 * - force and energy output with pair list pruning.
 */
#define FUNCTION_DECLARATION_ONLY
/** Force only **/
#include "nbnxm_cuda_kernels_hip.h"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxm_cuda_kernels_hip.h"
#undef CALC_ENERGIES

/*** Pair-list pruning kernels ***/
/** Force only **/
#define PRUNE_NBL
#include "nbnxm_cuda_kernels_hip.h"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxm_cuda_kernels_hip.h"
#undef CALC_ENERGIES
#undef PRUNE_NBL

/* Prune-only kernels */
#include "nbnxm_cuda_kernel_pruneonly_hip.h"
#undef FUNCTION_DECLARATION_ONLY

#define NTHREAD_Z_VALUE 4
#define FUNCTION_DECLARATION_ONLY
/** Force only **/
#include "nbnxm_cuda_kernels_hip.h"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxm_cuda_kernels_hip.h"
#undef CALC_ENERGIES

/*** Pair-list pruning kernels ***/
/** Force only **/
#define PRUNE_NBL
#include "nbnxm_cuda_kernels_hip.h"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxm_cuda_kernels_hip.h"
#undef CALC_ENERGIES
#undef PRUNE_NBL
#undef NTHREAD_Z_VALUE

/* Now generate the function definitions if we are using a single compilation unit. */
#if GMX_HIP_NB_SINGLE_COMPILATION_UNIT
#    include "nbnxm_cuda_kernel_F_noprune_hip.cpp"
#    include "nbnxm_cuda_kernel_F_prune_hip.cpp"
#    include "nbnxm_cuda_kernel_VF_noprune_hip.cpp"
#    include "nbnxm_cuda_kernel_VF_prune_hip.cpp"
#    include "nbnxm_cuda_kernel_pruneonly_hip.cpp"
#endif /* GMX_CUDA_NB_SINGLE_COMPILATION_UNIT */

namespace Nbnxm
{

/*! Nonbonded kernel function pointer type */
typedef void (*nbnxn_cu_kfunc_ptr_t)(const cu_atomdata_t, const NBParamGpu, const gpu_plist, bool);

/*********************************/

/*! Returns the number of blocks to be used for the nonbonded GPU kernel. */
static inline int calc_nb_kernel_nblock(int nwork_units, const DeviceInformation* deviceInfo)
{
    int max_grid_x_size;

    assert(deviceInfo);
    /* CUDA does not accept grid dimension of 0 (which can happen e.g. with an
       empty domain) and that case should be handled before this point. */
    assert(nwork_units > 0);

    max_grid_x_size = deviceInfo->prop.maxGridSize[0];

    /* do we exceed the grid x dimension limit? */
    if (nwork_units > max_grid_x_size)
    {
        gmx_fatal(FARGS,
                  "Watch out, the input system is too large to simulate!\n"
                  "The number of nonbonded work units (=number of super-clusters) exceeds the"
                  "maximum grid size in x dimension (%d > %d)!",
                  nwork_units, max_grid_x_size);
    }

    return nwork_units;
}


/* Constant arrays listing all kernel function pointers and enabling selection
   of a kernel in an elegant manner. */

/*! Pointers to the non-bonded kernels organized in 2-dim arrays by:
 *  electrostatics and VDW type.
 *
 *  Note that the row- and column-order of function pointers has to match the
 *  order of corresponding enumerated electrostatics and vdw types, resp.,
 *  defined in nbnxn_cuda_types.h.
 */

/*! Force-only kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_noener_noprune_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_cuda, nbnxn_kernel_ElecCut_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecCut_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_cuda, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecRF_VdwLJ_F_cuda, nbnxn_kernel_ElecRF_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecRF_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_cuda, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEw_VdwLJ_F_cuda, nbnxn_kernel_ElecEw_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_cuda }
};

static const nbnxn_cu_kfunc_ptr_t nb_kfunc_noener_noprune_dimZ_4_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecRF_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEw_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_cuda_dimZ_4 }
};

static const nbnxn_cu_kfunc_ptr_t nb_pack_kfunc_noener_noprune_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_cuda, nbnxn_kernel_ElecCut_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecCut_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_cuda, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecRF_VdwLJ_F_cuda, nbnxn_kernel_ElecRF_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecRF_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_cuda, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_cuda,
      nbnxn_pack_kernel_ElecEwQSTab_VdwLJCombLB_F_cuda, nbnxn_pack_kernel_ElecEwQSTab_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEw_VdwLJ_F_cuda, nbnxn_kernel_ElecEw_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_cuda },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_cuda }
};

static const nbnxn_cu_kfunc_ptr_t nb_pack_kfunc_noener_noprune_dimZ_4_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecRF_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_pack_kernel_ElecEwQSTab_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_pack_kernel_ElecEwQSTab_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEw_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_cuda_dimZ_4 }
};

/*! Force + energy kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_ener_noprune_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_VF_cuda, nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_cuda,
      nbnxn_kernel_ElecCut_VdwLJCombLB_VF_cuda, nbnxn_kernel_ElecCut_VdwLJFsw_VF_cuda,
      nbnxn_kernel_ElecCut_VdwLJPsw_VF_cuda, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_cuda,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_cuda },
    { nbnxn_kernel_ElecRF_VdwLJ_VF_cuda, nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_cuda,
      nbnxn_kernel_ElecRF_VdwLJCombLB_VF_cuda, nbnxn_kernel_ElecRF_VdwLJFsw_VF_cuda,
      nbnxn_kernel_ElecRF_VdwLJPsw_VF_cuda, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_cuda,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_cuda },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_cuda },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_cuda },
    { nbnxn_kernel_ElecEw_VdwLJ_VF_cuda, nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_cuda,
      nbnxn_kernel_ElecEw_VdwLJCombLB_VF_cuda, nbnxn_kernel_ElecEw_VdwLJFsw_VF_cuda,
      nbnxn_kernel_ElecEw_VdwLJPsw_VF_cuda, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_cuda,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_cuda },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_cuda }
};

static const nbnxn_cu_kfunc_ptr_t nb_kfunc_ener_noprune_dimZ_4_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_VF_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJCombLB_VF_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJFsw_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJPsw_VF_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_cuda_dimZ_4 },
    { nbnxn_kernel_ElecRF_VdwLJ_VF_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJCombLB_VF_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJFsw_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJPsw_VF_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEw_VdwLJ_VF_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJCombLB_VF_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJFsw_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJPsw_VF_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_cuda_dimZ_4 }
};

/*! Force + pruning kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_noener_prune_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_prune_cuda, nbnxn_kernel_ElecCut_VdwLJCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_prune_cuda, nbnxn_kernel_ElecCut_VdwLJFsw_F_prune_cuda,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_prune_cuda, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_prune_cuda },
    { nbnxn_kernel_ElecRF_VdwLJ_F_prune_cuda, nbnxn_kernel_ElecRF_VdwLJCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_prune_cuda, nbnxn_kernel_ElecRF_VdwLJFsw_F_prune_cuda,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_prune_cuda, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_prune_cuda },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_prune_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_prune_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_prune_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_prune_cuda },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_prune_cuda, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_prune_cuda },
    { nbnxn_kernel_ElecEw_VdwLJ_F_prune_cuda, nbnxn_kernel_ElecEw_VdwLJCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_prune_cuda, nbnxn_kernel_ElecEw_VdwLJFsw_F_prune_cuda,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_prune_cuda, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_prune_cuda },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_prune_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_prune_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_prune_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_prune_cuda }
};

static const nbnxn_cu_kfunc_ptr_t nb_kfunc_noener_prune_dimZ_4_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJFsw_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecRF_VdwLJ_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJFsw_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEw_VdwLJ_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJFsw_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_prune_cuda_dimZ_4 }
};

/*! Force + energy + pruning kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_ener_prune_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_VF_prune_cuda, nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecCut_VdwLJCombLB_VF_prune_cuda, nbnxn_kernel_ElecCut_VdwLJFsw_VF_prune_cuda,
      nbnxn_kernel_ElecCut_VdwLJPsw_VF_prune_cuda, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_prune_cuda },
    { nbnxn_kernel_ElecRF_VdwLJ_VF_prune_cuda, nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecRF_VdwLJCombLB_VF_prune_cuda, nbnxn_kernel_ElecRF_VdwLJFsw_VF_prune_cuda,
      nbnxn_kernel_ElecRF_VdwLJPsw_VF_prune_cuda, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_prune_cuda },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_prune_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_prune_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_prune_cuda, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_prune_cuda },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_prune_cuda },
    { nbnxn_kernel_ElecEw_VdwLJ_VF_prune_cuda, nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEw_VdwLJCombLB_VF_prune_cuda, nbnxn_kernel_ElecEw_VdwLJFsw_VF_prune_cuda,
      nbnxn_kernel_ElecEw_VdwLJPsw_VF_prune_cuda, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_prune_cuda },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_prune_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_prune_cuda, nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_prune_cuda,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_prune_cuda }
};

/*! Force + energy + pruning kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_ener_prune_dimZ_4_ptr[eelTypeNR][evdwTypeNR] = {
    { nbnxn_kernel_ElecCut_VdwLJ_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJCombLB_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJFsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJPsw_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecRF_VdwLJ_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJCombLB_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJFsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJPsw_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEw_VdwLJ_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJCombLB_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJFsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJPsw_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_prune_cuda_dimZ_4 },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_prune_cuda_dimZ_4, nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_prune_cuda_dimZ_4,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_prune_cuda_dimZ_4 }
};

/*! Return a pointer to the kernel version to be executed at the current step. */
static inline nbnxn_cu_kfunc_ptr_t select_nbnxn_kernel(int                     eeltype,
                                                       int                     evdwtype,
                                                       bool                    bDoEne,
                                                       bool                    bDoPrune,
                                                       const DeviceInformation gmx_unused* deviceInfo,
                                                       bool                    pack = false,
                                                       bool                    nthread4 = false)
{
    nbnxn_cu_kfunc_ptr_t res;

    GMX_ASSERT(eeltype < eelTypeNR,
               "The electrostatics type requested is not implemented in the HIP kernels.");
    GMX_ASSERT(evdwtype < evdwTypeNR,
               "The VdW type requested is not implemented in the HIP kernels.");

    /* assert assumptions made by the kernels */
    GMX_ASSERT(c_nbnxnGpuClusterSize * c_nbnxnGpuClusterSize / c_nbnxnGpuClusterpairSplit
                       == deviceInfo->prop.warpSize,
               "The HIP kernels require the "
               "cluster_size_i*cluster_size_j/nbnxn_gpu_clusterpair_split to match the warp size "
               "of the architecture targeted.");

    if (bDoEne)
    {
        if (bDoPrune)
        {
            if(nthread4)
                res = nb_kfunc_ener_prune_dimZ_4_ptr[eeltype][evdwtype];
            else
                res = nb_kfunc_ener_prune_ptr[eeltype][evdwtype];
        }
        else
        {
            if(nthread4)
                res = nb_kfunc_ener_noprune_dimZ_4_ptr[eeltype][evdwtype];
            else
                res = nb_kfunc_ener_noprune_ptr[eeltype][evdwtype];
        }
    }
    else
    {
        if (bDoPrune)
        {
            if(nthread4)
                res = nb_kfunc_noener_prune_dimZ_4_ptr[eeltype][evdwtype];
            else
                res = nb_kfunc_noener_prune_ptr[eeltype][evdwtype];
        }
        else
        {
            if (!pack) {
                if(nthread4)
                    res = nb_kfunc_noener_noprune_dimZ_4_ptr[eeltype][evdwtype];
                else
                    res = nb_kfunc_noener_noprune_ptr[eeltype][evdwtype];
	        } else {
                if(nthread4)
                    res = nb_pack_kfunc_noener_noprune_dimZ_4_ptr[eeltype][evdwtype];
                else
                    res = nb_pack_kfunc_noener_noprune_ptr[eeltype][evdwtype];
	        }
        }
    }

    return res;
}

/*! \brief Calculates the amount of shared memory required by the nonbonded kernel in use. */
static inline int calc_shmem_required_nonbonded(const int               num_threads_z,
                                                const DeviceInformation gmx_unused* deviceInfo,
                                                const NBParamGpu*                   nbp)
{
    int shmem;

    assert(deviceInfo);

    /* size of shmem (force-buffers/xq/atom type preloading) */
    /* NOTE: with the default kernel on sm3.0 we need shmem only for pre-loading */
    /* i-atom x+q in shared memory */
    shmem = c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(float4);
    /* cj in shared memory, for each warp separately */
    //shmem += num_threads_z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(int);

    if (nbp->vdwtype == evdwTypeCUTCOMBGEOM || nbp->vdwtype == evdwTypeCUTCOMBLB)
    {
        /* i-atom LJ combination parameters in shared memory */
        shmem += c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(float2);
    }
    else
    {
        /* i-atom types in shared memory */
        shmem += c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(int);
    }

    return shmem;
}

/*! \brief Sync the nonlocal stream with dependent tasks in the local queue.
 *
 *  As the point where the local stream tasks can be considered complete happens
 *  at the same call point where the nonlocal stream should be synced with the
 *  the local, this function records the event if called with the local stream as
 *  argument and inserts in the GPU stream a wait on the event on the nonlocal.
 */
void nbnxnInsertNonlocalGpuDependency(const NbnxmGpu* nb, const InteractionLocality interactionLocality)
{
    const DeviceStream& deviceStream = *nb->deviceStreams[interactionLocality];

    /* When we get here all misc operations issued in the local stream as well as
       the local xq H2D are done,
       so we record that in the local stream and wait for it in the nonlocal one.
       This wait needs to precede any PP tasks, bonded or nonbonded, that may
       compute on interactions between local and nonlocal atoms.
     */
    if (nb->bUseTwoStreams)
    {
        if (interactionLocality == InteractionLocality::Local)
        {
            hipError_t stat = hipEventRecord(nb->misc_ops_and_local_H2D_done, deviceStream.stream());
            CU_RET_ERR(stat, "hipEventRecord on misc_ops_and_local_H2D_done failed");
        }
        else
        {
            hipError_t stat =
                    hipStreamWaitEvent(deviceStream.stream(), nb->misc_ops_and_local_H2D_done, 0);
            CU_RET_ERR(stat, "hipStreamWaitEvent on misc_ops_and_local_H2D_done failed");
        }
    }
}

/*! \brief Launch asynchronously the xq buffer host to device copy. */
void gpu_copy_xq_to_gpu(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom, const AtomLocality atomLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    GMX_ASSERT(atomLocality == AtomLocality::Local || atomLocality == AtomLocality::NonLocal,
               "Only local and non-local xq transfers are supported");

    const InteractionLocality iloc = gpuAtomToInteractionLocality(atomLocality);

    int adat_begin, adat_len; /* local/nonlocal offset and length used for xq and f */

    cu_atomdata_t*      adat         = nb->atdat;
    gpu_plist*          plist        = nb->plist[iloc];
    cu_timers_t*        t            = nb->timers;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    bool bDoTime = nb->bDoTime;

    /* Don't launch the non-local H2D copy if there is no dependent
       work to do: neither non-local nor other (e.g. bonded) work
       to do that has as input the nbnxn coordaintes.
       Doing the same for the local kernel is more complicated, since the
       local part of the force array also depends on the non-local kernel.
       So to avoid complicating the code and to reduce the risk of bugs,
       we always call the local local x+q copy (and the rest of the local
       work in nbnxn_gpu_launch_kernel().
     */
    if ((iloc == InteractionLocality::NonLocal) && !haveGpuShortRangeWork(*nb, iloc))
    {
        plist->haveFreshList = false;

        return;
    }

    /* calculate the atom data index range based on locality */
    if (atomLocality == AtomLocality::Local)
    {
        adat_begin = 0;
        adat_len   = adat->natoms_local;
    }
    else
    {
        adat_begin = adat->natoms_local;
        adat_len   = adat->natoms - adat->natoms_local;
    }

    /* HtoD x, q */
    /* beginning of timed HtoD section */
    if (bDoTime)
    {
        t->xf[atomLocality].nb_h2d.openTimingRegion(deviceStream);
    }

    static_assert(sizeof(adat->xq[0]) == sizeof(float4),
                  "The size of the xyzq buffer element should be equal to the size of float4.");
    copyToDeviceBuffer(&adat->xq, reinterpret_cast<const float4*>(nbatom->x().data()) + adat_begin,
                       adat_begin, adat_len, deviceStream, GpuApiCallBehavior::Async, nullptr);

    if (bDoTime)
    {
        t->xf[atomLocality].nb_h2d.closeTimingRegion(deviceStream);
    }

    /* When we get here all misc operations issued in the local stream as well as
       the local xq H2D are done,
       so we record that in the local stream and wait for it in the nonlocal one.
       This wait needs to precede any PP tasks, bonded or nonbonded, that may
       compute on interactions between local and nonlocal atoms.
     */
    nbnxnInsertNonlocalGpuDependency(nb, iloc);
}

/*! As we execute nonbonded workload in separate streams, before launching
   the kernel we need to make sure that he following operations have completed:
   - atomdata allocation and related H2D transfers (every nstlist step);
   - pair list H2D transfer (every nstlist step);
   - shift vector H2D transfer (every nstlist step);
   - force (+shift force and energy) output clearing (every step).

   These operations are issued in the local stream at the beginning of the step
   and therefore always complete before the local kernel launch. The non-local
   kernel is launched after the local on the same device/context hence it is
   inherently scheduled after the operations in the local stream (including the
   above "misc_ops") on pre-GK110 devices with single hardware queue, but on later
   devices with multiple hardware queues the dependency needs to be enforced.
   We use the misc_ops_and_local_H2D_done event to record the point where
   the local x+q H2D (and all preceding) tasks are complete and synchronize
   with this event in the non-local stream before launching the non-bonded kernel.
 */
void gpu_launch_kernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc)
{
    cu_atomdata_t*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    cu_timers_t*        t            = nb->timers;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    bool bDoTime = nb->bDoTime;

    /* Don't launch the non-local kernel if there is no work to do.
       Doing the same for the local kernel is more complicated, since the
       local part of the force array also depends on the non-local kernel.
       So to avoid complicating the code and to reduce the risk of bugs,
       we always call the local kernel, and later (not in
       this function) the stream wait, local f copyback and the f buffer
       clearing. All these operations, except for the local interaction kernel,
       are needed for the non-local interactions. The skip of the local kernel
       call is taken care of later in this function. */
    if (canSkipNonbondedWork(*nb, iloc))
    {
        plist->haveFreshList = false;

        return;
    }

    if (nbp->useDynamicPruning && plist->haveFreshList)
    {
        /* Prunes for rlistOuter and rlistInner, sets plist->haveFreshList=false
           (TODO: ATM that's the way the timing accounting can distinguish between
           separate prune kernel and combined force+prune, maybe we need a better way?).
         */
        gpu_launch_kernel_pruneonly(nb, iloc, 1);
    }

    if (plist->nsci == 0)
    {
        /* Don't launch an empty local kernel (not allowed with CUDA) */
        return;
    }

    /* beginning of timed nonbonded calculation section */
    if (bDoTime)
    {
        t->interaction[iloc].nb_k.openTimingRegion(deviceStream);
    }

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    int num_threads_z = 1;
    if (nb->deviceContext_->deviceInfo().prop.gcnArch > 908)
    {
        num_threads_z = 4;
    }
    int nblock = calc_nb_kernel_nblock(plist->nsci, &nb->deviceContext_->deviceInfo());


    KernelLaunchConfig config;
    config.blockSize[0] = c_clSize;
    config.blockSize[1] = c_clSize;
    config.blockSize[2] = num_threads_z;
    config.gridSize[0]  = nblock;
    config.sharedMemorySize =
            calc_shmem_required_nonbonded(num_threads_z, &nb->deviceContext_->deviceInfo(), nbp);

    if (debug)
    {
        fprintf(debug,
                "Non-bonded GPU launch configuration:\n\tThread block: %zux%zux%zu\n\t"
                "\tGrid: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
                "\tShMem: %zu\n",
                config.blockSize[0], config.blockSize[1], config.blockSize[2], config.gridSize[0],
                config.gridSize[1], plist->nsci * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster, plist->na_c, config.sharedMemorySize);
    }

    bool usePack = false;

#ifdef GMX_GPU_USE_PACK
    //if (nbp->eeltype == 2 && nbp->vdwtype == 3 &&
    //    	    !stepWork.computeEnergy &&
    //    	    !(plist->haveFreshList &&
    //                !nb->timers->interaction[iloc].didPrune)) {
    //    usePack = true;
    //}
    usePack = true;
#endif

    auto*      timingEvent = bDoTime ? t->interaction[iloc].nb_k.fetchNextEvent() : nullptr;
    const auto kernel =
            select_nbnxn_kernel(nbp->eeltype, nbp->vdwtype, stepWork.computeEnergy,
                                (plist->haveFreshList && !nb->timers->interaction[iloc].didPrune),
                                &nb->deviceContext_->deviceInfo(), usePack, num_threads_z == 4);
    /*
    const auto kernelArgs =
            prepareGpuKernelArguments(kernel, config, adat, nbp, plist, &stepWork.computeVirial);
    launchGpuKernel(kernel, config, deviceStream, timingEvent, "k_calc_nb", kernelArgs);
    */
    launchGpuKernel(kernel, config, deviceStream, timingEvent, "k_calc_nb", *adat, *nbp, *plist, stepWork.computeVirial);

    if (bDoTime)
    {
        t->interaction[iloc].nb_k.closeTimingRegion(deviceStream);
    }

    if (GMX_NATIVE_WINDOWS)
    {
        /* Windows: force flushing WDDM queue */
        hipStreamQuery(deviceStream.stream());
    }
}

/*! Calculates the amount of shared memory required by the CUDA kernel in use. */
static inline int calc_shmem_required_prune(const int num_threads_z)
{
    int shmem;

    /* i-atom x in shared memory */
    shmem = c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(float4);
    /* cj in shared memory, for each warp separately */
    //shmem += num_threads_z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(int);

    return shmem;
}

void gpu_launch_kernel_pruneonly(NbnxmGpu* nb, const InteractionLocality iloc, const int numParts)
{
    cu_atomdata_t*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    cu_timers_t*        t            = nb->timers;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    bool bDoTime = nb->bDoTime;

    if (plist->haveFreshList)
    {
        GMX_ASSERT(numParts == 1, "With first pruning we expect 1 part");

        /* Set rollingPruningNumParts to signal that it is not set */
        plist->rollingPruningNumParts = 0;
        plist->rollingPruningPart     = 0;
    }
    else
    {
        if (plist->rollingPruningNumParts == 0)
        {
            plist->rollingPruningNumParts = numParts;
        }
        else
        {
            GMX_ASSERT(numParts == plist->rollingPruningNumParts,
                       "It is not allowed to change numParts in between list generation steps");
        }
    }

    /* Use a local variable for part and update in plist, so we can return here
     * without duplicating the part increment code.
     */
    int part = plist->rollingPruningPart;

    plist->rollingPruningPart++;
    if (plist->rollingPruningPart >= plist->rollingPruningNumParts)
    {
        plist->rollingPruningPart = 0;
    }

    /* Compute the number of list entries to prune in this pass */
    int numSciInPart = (plist->nsci - part) / numParts;

    /* Don't launch the kernel if there is no work to do (not allowed with CUDA) */
    if (numSciInPart <= 0)
    {
        plist->haveFreshList = false;

        return;
    }

    GpuRegionTimer* timer = nullptr;
    if (bDoTime)
    {
        timer = &(plist->haveFreshList ? t->interaction[iloc].prune_k : t->interaction[iloc].rollingPrune_k);
    }

    /* beginning of timed prune calculation section */
    if (bDoTime)
    {
        timer->openTimingRegion(deviceStream);
    }

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    int num_threads_z = c_cudaPruneKernelJ4Concurrency;
    int nblock        = calc_nb_kernel_nblock(numSciInPart, &nb->deviceContext_->deviceInfo());
    KernelLaunchConfig config;
    config.blockSize[0]     = c_clSize;
    config.blockSize[1]     = c_clSize;
    config.blockSize[2]     = num_threads_z;
    config.gridSize[0]      = nblock;
    config.sharedMemorySize = calc_shmem_required_prune(num_threads_z);

    if (debug)
    {
        fprintf(debug,
                "Pruning GPU kernel launch configuration:\n\tThread block: %zux%zux%zu\n\t"
                "\tGrid: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
                "\tShMem: %zu\n",
                config.blockSize[0], config.blockSize[1], config.blockSize[2], config.gridSize[0],
                config.gridSize[1], numSciInPart * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster, plist->na_c, config.sharedMemorySize);
    }

    auto*          timingEvent  = bDoTime ? timer->fetchNextEvent() : nullptr;
    constexpr char kernelName[] = "k_pruneonly";
    const auto     kernel =
            plist->haveFreshList ? nbnxn_kernel_prune_cuda<true> : nbnxn_kernel_prune_cuda<false>;
    /*
    const auto kernelArgs = prepareGpuKernelArguments(kernel, config, adat, nbp, plist, &numParts, &part);
    launchGpuKernel(kernel, config, deviceStream, timingEvent, kernelName, kernelArgs);
    */
    launchGpuKernel(kernel, config, deviceStream, timingEvent, kernelName, *adat, *nbp, *plist, numParts, part);

    /* TODO: consider a more elegant way to track which kernel has been called
       (combined or separate 1st pass prune, rolling prune). */
    if (plist->haveFreshList)
    {
        plist->haveFreshList = false;
        /* Mark that pruning has been done */
        nb->timers->interaction[iloc].didPrune = true;
    }
    else
    {
        /* Mark that rolling pruning has been done */
        nb->timers->interaction[iloc].didRollingPrune = true;
    }

    if (bDoTime)
    {
        timer->closeTimingRegion(deviceStream);
    }

    if (GMX_NATIVE_WINDOWS)
    {
        /* Windows: force flushing WDDM queue */
        hipStreamQuery(deviceStream.stream());
    }
}

void gpu_launch_cpyback(NbnxmGpu*                nb,
                        nbnxn_atomdata_t*        nbatom,
                        const gmx::StepWorkload& stepWork,
                        const AtomLocality       atomLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    hipError_t stat;
    int         adat_begin, adat_len; /* local/nonlocal offset and length used for xq and f */

    /* determine interaction locality from atom locality */
    const InteractionLocality iloc = gpuAtomToInteractionLocality(atomLocality);

    /* extract the data */
    cu_atomdata_t*      adat         = nb->atdat;
    cu_timers_t*        t            = nb->timers;
    bool                bDoTime      = nb->bDoTime;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    /* don't launch non-local copy-back if there was no non-local work to do */
    if ((iloc == InteractionLocality::NonLocal) && !haveGpuShortRangeWork(*nb, iloc))
    {
        return;
    }

    getGpuAtomRange(adat, atomLocality, &adat_begin, &adat_len);

    /* beginning of timed D2H section */
    if (bDoTime)
    {
        t->xf[atomLocality].nb_d2h.openTimingRegion(deviceStream);
    }

    /* With DD the local D2H transfer can only start after the non-local
       kernel has finished. */
    if (iloc == InteractionLocality::Local && nb->bUseTwoStreams)
    {
        stat = hipStreamWaitEvent(deviceStream.stream(), nb->nonlocal_done, 0);
        CU_RET_ERR(stat, "hipStreamWaitEvent on nonlocal_done failed");
    }

    /* DtoH f
     * Skip if buffer ops / reduction is offloaded to the GPU.
     */
    if (!stepWork.useGpuFBufferOps)
    {
        static_assert(
                sizeof(adat->f[0]) == sizeof(float3),
                "The size of the force buffer element should be equal to the size of float3.");
        copyFromDeviceBuffer(reinterpret_cast<float3*>(nbatom->out[0].f.data()) + adat_begin, &adat->f,
                             adat_begin, adat_len, deviceStream, GpuApiCallBehavior::Async, nullptr);
    }

    /* After the non-local D2H is launched the nonlocal_done event can be
       recorded which signals that the local D2H can proceed. This event is not
       placed after the non-local kernel because we want the non-local data
       back first. */
    if (iloc == InteractionLocality::NonLocal)
    {
        stat = hipEventRecord(nb->nonlocal_done, deviceStream.stream());
        CU_RET_ERR(stat, "hipEventRecord on nonlocal_done failed");
    }

    /* only transfer energies in the local stream */
    if (iloc == InteractionLocality::Local)
    {
        /* DtoH fshift when virial is needed */
        if ( (stepWork.computeVirial && c_clShiftMemoryMultiplier != 0) ||
             (stepWork.computeEnergy && c_clEnergyMemoryMultiplier != 0) )
        {
            constexpr unsigned int block_size = 64U;

            KernelLaunchConfig configSumUp;
            configSumUp.blockSize[0] = block_size;
            configSumUp.blockSize[1] = 1;
            configSumUp.blockSize[2] = 1;
            configSumUp.gridSize[0]  = (stepWork.computeVirial && c_clShiftMemoryMultiplier != 0) ? SHIFTS : 1;
            configSumUp.sharedMemorySize = 0;

            const auto kernelSumUp = nbnxn_kernel_sum_up<block_size>;

            launchGpuKernel(
                kernelSumUp,
                configSumUp,
                deviceStream,
                nullptr,
                "nbnxn_kernel_sum_up",
                *adat,
                SHIFTS,
                (stepWork.computeEnergy && c_clEnergyMemoryMultiplier != 0),
                (stepWork.computeVirial && c_clShiftMemoryMultiplier != 0)
            );
        }

        if (stepWork.computeVirial)
        {
            static_assert(sizeof(nb->nbst.fshift[0]) == sizeof(adat->fshift[0]),
                          "Sizes of host- and device-side shift vectors should be the same.");
            copyFromDeviceBuffer(nb->nbst.fshift, &adat->fshift, 0, SHIFTS, deviceStream,
                                 GpuApiCallBehavior::Async, nullptr);
        }

        /* DtoH energies */
        if (stepWork.computeEnergy)
        {
            static_assert(sizeof(nb->nbst.e_lj[0]) == sizeof(adat->e_lj[0]),
                          "Sizes of host- and device-side LJ energy terms should be the same.");
            copyFromDeviceBuffer(nb->nbst.e_lj, &adat->e_lj, 0, 1, deviceStream,
                                 GpuApiCallBehavior::Async, nullptr);
            static_assert(sizeof(nb->nbst.e_el[0]) == sizeof(adat->e_el[0]),
                          "Sizes of host- and device-side electrostatic energy terms should be the "
                          "same.");
            copyFromDeviceBuffer(nb->nbst.e_el, &adat->e_el, 0, 1, deviceStream,
                                 GpuApiCallBehavior::Async, nullptr);
        }
    }

    if (bDoTime)
    {
        t->xf[atomLocality].nb_d2h.closeTimingRegion(deviceStream);
    }
}

void cuda_set_cacheconfig()
{
    hipError_t stat;

    for (int i = 0; i < eelTypeNR; i++)
    {
        for (int j = 0; j < evdwTypeNR; j++)
        {
            /* Default kernel 32/32 kB Shared/L1 */
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_ener_prune_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_ener_prune_dimZ_4_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_ener_noprune_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_ener_noprune_dimZ_4_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_noener_prune_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_noener_prune_dimZ_4_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_pack_kfunc_noener_noprune_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_pack_kfunc_noener_noprune_dimZ_4_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_noener_noprune_ptr[i][j]), hipFuncCachePreferEqual);
            stat = hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_noener_noprune_dimZ_4_ptr[i][j]), hipFuncCachePreferEqual);
            CU_RET_ERR(stat, "hipFuncSetCacheConfig failed");
        }
    }
}

/* X buffer operations on GPU: performs conversion from rvec to nb format. */
void nbnxn_gpu_x_to_nbat_x(const Nbnxm::Grid&        grid,
                           bool                      setFillerCoords,
                           NbnxmGpu*                 nb,
                           DeviceBuffer<gmx::RVec>   d_x,
                           GpuEventSynchronizer*     xReadyOnDevice,
                           const Nbnxm::AtomLocality locality,
                           int                       gridId,
                           int                       numColumnsMax)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");

    cu_atomdata_t* adat = nb->atdat;

    const int                  numColumns      = grid.numColumns();
    const int                  cellOffset      = grid.cellOffset();
    const int                  numAtomsPerCell = grid.numAtomsPerCell();
    Nbnxm::InteractionLocality interactionLoc  = gpuAtomToInteractionLocality(locality);

    const DeviceStream& deviceStream = *nb->deviceStreams[interactionLoc];

    int numAtoms = grid.srcAtomEnd() - grid.srcAtomBegin();
    // avoid empty kernel launch, skip to inserting stream dependency
    if (numAtoms != 0)
    {
        // TODO: This will only work with CUDA
        GMX_ASSERT(d_x, "Need a valid device pointer");

        // ensure that coordinates are ready on the device before launching the kernel
        GMX_ASSERT(xReadyOnDevice, "Need a valid GpuEventSynchronizer object");
        xReadyOnDevice->enqueueWaitEvent(deviceStream);

        KernelLaunchConfig config;
        config.blockSize[0] = c_bufOpsThreadsPerBlock;
        config.blockSize[1] = 1;
        config.blockSize[2] = 1;
        config.gridSize[0] = (grid.numCellsColumnMax() * numAtomsPerCell + c_bufOpsThreadsPerBlock - 1)
                             / c_bufOpsThreadsPerBlock;
        config.gridSize[1] = numColumns;
        config.gridSize[2] = 1;
        GMX_ASSERT(config.gridSize[0] > 0,
                   "Can not have empty grid, early return above avoids this");
        config.sharedMemorySize = 0;

        auto kernelFn = setFillerCoords ? nbnxn_gpu_x_to_nbat_x_kernel<true>
                                        : nbnxn_gpu_x_to_nbat_x_kernel<false>;
        float4*    d_xq          = adat->xq;
        float3*    d_xFloat3     = asFloat3(d_x);
        const int* d_atomIndices = nb->atomIndices;
        const int* d_cxy_na      = &nb->cxy_na[numColumnsMax * gridId];
        const int* d_cxy_ind     = &nb->cxy_ind[numColumnsMax * gridId];
	/*
        const auto kernelArgs    = prepareGpuKernelArguments(kernelFn,
                                                          config,
                                                          &numColumns,
                                                          &d_xq,
                                                          &d_xFloat3,
                                                          &d_atomIndices,
                                                          &d_cxy_na,
                                                          &d_cxy_ind,
                                                          &cellOffset,
                                                          &numAtomsPerCell);
        launchGpuKernel(kernelFn, config, deviceStream, nullptr, "XbufferOps", kernelArgs);
        */
        launchGpuKernel(kernelFn, config, deviceStream, nullptr, "XbufferOps",
                        numColumns, d_xq, const_cast<const float3*>(reinterpret_cast<float3*>(d_xFloat3)),
                        d_atomIndices, d_cxy_na, d_cxy_ind, cellOffset, numAtomsPerCell);
    }

    // TODO: note that this is not necessary when there astreamre no local atoms, that is:
    // (numAtoms == 0 && interactionLoc == InteractionLocality::Local)
    // but for now we avoid that optimization
    nbnxnInsertNonlocalGpuDependency(nb, interactionLoc);
}

void* getGpuForces(NbnxmGpu* nb)
{
    return nb->atdat->f;
}

} // namespace Nbnxm
