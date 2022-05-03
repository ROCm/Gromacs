/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
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
 *  \brief Define HIP implementation of nbnxn_gpu.h
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


#include "nbnxm_hip.h"

#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/typecasts.hpp"
#include "gromacs/gpu_utils/vectype_ops.hpp"
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

#include "nbnxm_hip_types.h"
#include <fstream>

#include <rocprim/rocprim.hpp>

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
#include "nbnxm_hip_kernels.hpp"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxm_hip_kernels.hpp"
#undef CALC_ENERGIES

/*** Pair-list pruning kernels ***/
/** Force only **/
#define PRUNE_NBL
#include "nbnxm_hip_kernels.hpp"
/** Force & energy **/
#define CALC_ENERGIES
#include "nbnxm_hip_kernels.hpp"
#undef CALC_ENERGIES
#undef PRUNE_NBL

/* Prune-only kernels */
#include "nbnxm_hip_kernel_pruneonly.hpp"
#undef FUNCTION_DECLARATION_ONLY

/* Now generate the function definitions if we are using a single compilation unit. */
#if GMX_HIP_NB_SINGLE_COMPILATION_UNIT
#    include "nbnxm_hip_kernel_F_noprune.hip.cpp"
#    include "nbnxm_hip_kernel_F_prune.hip.cpp"
#    include "nbnxm_hip_kernel_VF_noprune.hip.cpp"
#    include "nbnxm_hip_kernel_VF_prune.hip.cpp"
#    include "nbnxm_hip_kernel_pruneonly.hip.cpp"
#endif /* GMX_HIP_NB_SINGLE_COMPILATION_UNIT */

namespace Nbnxm
{

/*! Nonbonded kernel function pointer type */
typedef void (*nbnxn_cu_kfunc_ptr_t)(const NBAtomDataGpu, const NBParamGpu, const gpu_plist, bool);

/*********************************/

/*! Returns the number of blocks to be used for the nonbonded GPU kernel. */
static inline int calc_nb_kernel_nblock(int nwork_units, const DeviceInformation* deviceInfo)
{
    int max_grid_x_size;

    assert(deviceInfo);
    /* HIP does not accept grid dimension of 0 (which can happen e.g. with an
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
                  nwork_units,
                  max_grid_x_size);
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
 *  defined in nbnxn_hip_types.h.
 */

/*! Force-only kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_noener_noprune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_hip,
      nbnxn_kernel_ElecCut_VdwLJCombGeom_F_hip,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_hip,
      nbnxn_kernel_ElecCut_VdwLJFsw_F_hip,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_hip },
    { nbnxn_kernel_ElecRF_VdwLJ_F_hip,
      nbnxn_kernel_ElecRF_VdwLJCombGeom_F_hip,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_hip,
      nbnxn_kernel_ElecRF_VdwLJFsw_F_hip,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_hip },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_hip },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_hip },
    { nbnxn_kernel_ElecEw_VdwLJ_F_hip,
      nbnxn_kernel_ElecEw_VdwLJCombGeom_F_hip,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_hip,
      nbnxn_kernel_ElecEw_VdwLJFsw_F_hip,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_hip },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_hip }
};

/*! Force + energy kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_ener_noprune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { nbnxn_kernel_ElecCut_VdwLJ_VF_hip,
      nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_hip,
      nbnxn_kernel_ElecCut_VdwLJCombLB_VF_hip,
      nbnxn_kernel_ElecCut_VdwLJFsw_VF_hip,
      nbnxn_kernel_ElecCut_VdwLJPsw_VF_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_hip },
    { nbnxn_kernel_ElecRF_VdwLJ_VF_hip,
      nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_hip,
      nbnxn_kernel_ElecRF_VdwLJCombLB_VF_hip,
      nbnxn_kernel_ElecRF_VdwLJFsw_VF_hip,
      nbnxn_kernel_ElecRF_VdwLJPsw_VF_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_hip },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_hip },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_hip },
    { nbnxn_kernel_ElecEw_VdwLJ_VF_hip,
      nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_hip,
      nbnxn_kernel_ElecEw_VdwLJCombLB_VF_hip,
      nbnxn_kernel_ElecEw_VdwLJFsw_VF_hip,
      nbnxn_kernel_ElecEw_VdwLJPsw_VF_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_hip },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_hip }
};

/*! Force + pruning kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_noener_prune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { nbnxn_kernel_ElecCut_VdwLJ_F_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJCombGeom_F_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJCombLB_F_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJFsw_F_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJPsw_F_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_prune_hip },
    { nbnxn_kernel_ElecRF_VdwLJ_F_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJCombGeom_F_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJCombLB_F_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJFsw_F_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJPsw_F_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_prune_hip },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_F_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_F_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_prune_hip },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_F_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_prune_hip },
    { nbnxn_kernel_ElecEw_VdwLJ_F_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJCombLB_F_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJFsw_F_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJPsw_F_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_prune_hip },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_F_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_prune_hip }
};

/*! Force + energy + pruning kernel function pointers. */
static const nbnxn_cu_kfunc_ptr_t nb_kfunc_ener_prune_ptr[c_numElecTypes][c_numVdwTypes] = {
    { nbnxn_kernel_ElecCut_VdwLJ_VF_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJCombLB_VF_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJFsw_VF_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJPsw_VF_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_prune_hip },
    { nbnxn_kernel_ElecRF_VdwLJ_VF_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJCombLB_VF_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJFsw_VF_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJPsw_VF_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_prune_hip },
    { nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJCombLB_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_prune_hip },
    { nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJCombLB_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_prune_hip },
    { nbnxn_kernel_ElecEw_VdwLJ_VF_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJCombLB_VF_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJFsw_VF_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJPsw_VF_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_prune_hip },
    { nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJCombLB_VF_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_prune_hip,
      nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_prune_hip }
};

/*! Return a pointer to the kernel version to be executed at the current step. */
static inline nbnxn_cu_kfunc_ptr_t select_nbnxn_kernel(enum ElecType           elecType,
                                                       enum VdwType            vdwType,
                                                       bool                    bDoEne,
                                                       bool                    bDoPrune,
                                                       const DeviceInformation gmx_unused* deviceInfo)
{
    const int elecTypeIdx = static_cast<int>(elecType);
    const int vdwTypeIdx  = static_cast<int>(vdwType);

    GMX_ASSERT(elecTypeIdx < c_numElecTypes,
               "The electrostatics type requested is not implemented in the HIP kernels.");
    GMX_ASSERT(vdwTypeIdx < c_numVdwTypes,
               "The VdW type requested is not implemented in the HIP kernels.");

    /* assert assumptions made by the kernels */
    GMX_ASSERT(deviceInfo->prop.warpSize % (c_nbnxnGpuClusterSize * c_nbnxnGpuClusterSize / c_nbnxnGpuClusterpairSplit)  == 0,
               "The HIP kernels require the "
               "cluster_size_i*cluster_size_j/nbnxn_gpu_clusterpair_split to be dividable with the warp size "
               "of the architecture targeted.");

    if (bDoEne)
    {
        if (bDoPrune)
        {
            return nb_kfunc_ener_prune_ptr[elecTypeIdx][vdwTypeIdx];
        }
        else
        {
            return nb_kfunc_ener_noprune_ptr[elecTypeIdx][vdwTypeIdx];
        }
    }
    else
    {
        if (bDoPrune)
        {
            return nb_kfunc_noener_prune_ptr[elecTypeIdx][vdwTypeIdx];
        }
        else
        {
            return nb_kfunc_noener_noprune_ptr[elecTypeIdx][vdwTypeIdx];
        }
    }
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

    if (nbp->vdwType == VdwType::CutCombGeom || nbp->vdwType == VdwType::CutCombLB)
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
    NBAtomDataGpu*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    Nbnxm::GpuTimers*   timers       = nb->timers;
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

    /*{
        std::vector<nbnxn_sci_t> host_sci(plist->nsci);

        hipError_t  stat = hipMemcpy(host_sci.data(),
                                     *reinterpret_cast<nbnxn_sci_t**>(&(plist->sci)),
                                     plist->nsci * sizeof(nbnxn_sci_t),
                                     hipMemcpyDeviceToHost);

        std::ofstream scifile;
        scifile.open("sci_before_prune.out", std::ios::app);
        scifile << "---------------------START-------------------- " << stat << std::endl;
        for(unsigned int index_sci = 0; index_sci < plist->nsci; index_sci++)
        {
            scifile << host_sci[index_sci].sci << " ; " << host_sci[index_sci].cj4_ind_start << " ; ";
            scifile << host_sci[index_sci].cj4_length << " ; " << host_sci[index_sci].shift << std::endl;
        }
        scifile << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
        scifile.close();

        std::vector<nbnxn_cj4_t> host_cj4(plist->ncj4);
        stat = hipMemcpy(host_cj4.data(),
                         *reinterpret_cast<nbnxn_cj4_t**>(&(plist->cj4)),
                         plist->ncj4 * sizeof(nbnxn_cj4_t),
                         hipMemcpyDeviceToHost);

        unsigned int cj4mask_histogram[32];
        for(unsigned int index_histogram = 0; index_histogram < 32; index_histogram++)
        {
            cj4mask_histogram[index_histogram] = 0;
        }

        std::ofstream cj4file;
        cj4file.open("cj4_before_prune.out", std::ios::app);
        cj4file << "---------------------START-------------------- " << stat << std::endl;

        std::ofstream countfile;
        countfile.open("countfile_prune.out", std::ios::app);
        countfile << "---------------------START-------------------- " << std::endl;

        for(unsigned int index_sci = 0; index_sci < plist->nsci; index_sci++)
        {
            unsigned int count = 0;
            for (int j4 = host_sci[index_sci].cj4_ind_start; j4 < host_sci[index_sci].cj4IndEnd(); j4++)
            {
                cj4file << host_cj4[j4].cj[0] << " ; " << host_cj4[j4].cj[1] << " ; ";
                cj4file << host_cj4[j4].cj[2] << " ; " << host_cj4[j4].cj[3] << " ; ";
                cj4file << host_cj4[j4].imei[0].imask << " ; " <<  host_cj4[j4].imei[1].imask << " ; ";
                cj4file << __builtin_popcount(host_cj4[j4].imei[0].imask & host_cj4[j4].imei[1].imask) << " ; " << __builtin_popcount(host_cj4[j4].imei[0].imask | host_cj4[j4].imei[1].imask) << " ; ";
                cj4file << host_cj4[j4].imei[0].excl_ind << " ; " <<  host_cj4[j4].imei[1].excl_ind << " ; ";
                cj4file << std::endl;

                count += __builtin_popcount(host_cj4[j4].imei[0].imask | host_cj4[j4].imei[1].imask);

                cj4mask_histogram[__builtin_popcount(host_cj4[j4].imei[0].imask | host_cj4[j4].imei[1].imask) - __builtin_popcount(host_cj4[j4].imei[0].imask & host_cj4[j4].imei[1].imask)]++;
            }
            countfile << index_sci << " , " << count << std::endl;
            cj4file << std::endl;
        }
        cj4file << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
        cj4file.close();
        countfile.close();

        std::ofstream cj4mask_histogram_file;
        cj4mask_histogram_file.open("cj4_histogram_before_prune.out", std::ios::app);
        cj4mask_histogram_file << "---------------------START-------------------- " << std::endl;

        for(unsigned int index_histogram = 0; index_histogram < 32; index_histogram++)
        {
            cj4mask_histogram_file << cj4mask_histogram[index_histogram] << " ; ";
        }
        cj4mask_histogram_file.close();
    }*/

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
        /* Don't launch an empty local kernel (not allowed with HIP) */
        return;
    }

    /*{
        std::vector<nbnxn_sci_t> host_sci(plist->nsci);

        hipError_t  stat = hipMemcpy(host_sci.data(),
                                     *reinterpret_cast<nbnxn_sci_t**>(&(plist->sci_sorted)),
                                     plist->nsci * sizeof(nbnxn_sci_t),
                                     hipMemcpyDeviceToHost);

        std::ofstream scifile;
        scifile.open("sci.out", std::ios::app);
        scifile << "---------------------START-------------------- " << stat << std::endl;
        for(unsigned int index_sci = 0; index_sci < plist->nsci; index_sci++)
        {
            scifile << host_sci[index_sci].sci << " ; " << host_sci[index_sci].cj4_ind_start << " ; ";
            scifile << host_sci[index_sci].cj4_length << " ; " << host_sci[index_sci].shift << std::endl;
        }
        scifile << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
        scifile.close();

        std::vector<nbnxn_cj4_t> host_cj4(plist->ncj4);
        stat = hipMemcpy(host_cj4.data(),
                         *reinterpret_cast<nbnxn_cj4_t**>(&(plist->cj4)),
                         plist->ncj4 * sizeof(nbnxn_cj4_t),
                         hipMemcpyDeviceToHost);

        unsigned int cj4mask_histogram[32];
        for(unsigned int index_histogram = 0; index_histogram < 32; index_histogram++)
        {
            cj4mask_histogram[index_histogram] = 0;
        }

        std::ofstream cj4file;
        cj4file.open("cj4.out", std::ios::app);
        cj4file << "---------------------START-------------------- " << stat << std::endl;

        std::ofstream countfile;
        countfile.open("countfile.out", std::ios::app);
        countfile << "---------------------START-------------------- " << std::endl;

        for(unsigned int index_sci = 0; index_sci < plist->nsci; index_sci++)
        {
            unsigned int count = 0;
            for (int j4 = host_sci[index_sci].cj4_ind_start; j4 < host_sci[index_sci].cj4IndEnd(); j4++)
            {
                cj4file << host_cj4[j4].cj[0] << " ; " << host_cj4[j4].cj[1] << " ; ";
                cj4file << host_cj4[j4].cj[2] << " ; " << host_cj4[j4].cj[3] << " ; ";
                cj4file << host_cj4[j4].imei[0].imask << " ; " <<  host_cj4[j4].imei[1].imask << " ; ";
                cj4file << __builtin_popcount(host_cj4[j4].imei[0].imask & host_cj4[j4].imei[1].imask) << " ; " << __builtin_popcount(host_cj4[j4].imei[0].imask | host_cj4[j4].imei[1].imask) << " ; ";
                cj4file << host_cj4[j4].imei[0].excl_ind << " ; " <<  host_cj4[j4].imei[1].excl_ind << " ; ";
                cj4file << std::endl;

                count += __builtin_popcount(host_cj4[j4].imei[0].imask | host_cj4[j4].imei[1].imask);

                cj4mask_histogram[__builtin_popcount(host_cj4[j4].imei[0].imask | host_cj4[j4].imei[1].imask) - __builtin_popcount(host_cj4[j4].imei[0].imask & host_cj4[j4].imei[1].imask)]++;
            }
            countfile << index_sci << " , " << count << std::endl;
            cj4file << std::endl;
        }
        cj4file << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
        cj4file.close();
        countfile.close();

        std::ofstream cj4mask_histogram_file;
        cj4mask_histogram_file.open("cj4_histogram.out", std::ios::app);
        cj4mask_histogram_file << "---------------------START-------------------- " << std::endl;

        for(unsigned int index_histogram = 0; index_histogram < 32; index_histogram++)
        {
            cj4mask_histogram_file << cj4mask_histogram[index_histogram] << " ; ";
        }
        cj4mask_histogram_file.close();
    }*/

    /* beginning of timed nonbonded calculation section */
    if (bDoTime)
    {
        timers->interaction[iloc].nb_k.openTimingRegion(deviceStream);
    }

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    int num_threads_z = 2;
    int nblock = calc_nb_kernel_nblock(plist->nsci, &nb->deviceContext_->deviceInfo());


    KernelLaunchConfig config;
    config.blockSize[0] = c_clSize * c_clSize * num_threads_z;
    config.gridSize[0]  = nblock;
    config.sharedMemorySize =
            calc_shmem_required_nonbonded(num_threads_z, &nb->deviceContext_->deviceInfo(), nbp);

    if (debug)
    {
        fprintf(debug,
                "Non-bonded GPU launch configuration:\n\tThread block: %zux%zux%zu\n\t"
                "\tGrid: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
                "\tShMem: %zu\n",
                config.blockSize[0],
                config.blockSize[1],
                config.blockSize[2],
                config.gridSize[0],
                config.gridSize[1],
                plist->nsci * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster,
                plist->na_c,
                config.sharedMemorySize);
    }

    auto*      timingEvent = bDoTime ? timers->interaction[iloc].nb_k.fetchNextEvent() : nullptr;
    const auto kernel =
            select_nbnxn_kernel(nbp->elecType,
                                nbp->vdwType,
                                stepWork.computeEnergy,
                                (plist->haveFreshList && !nb->timers->interaction[iloc].didPrune),
                                &nb->deviceContext_->deviceInfo());
    const auto kernelArgs =
            prepareGpuKernelArguments(kernel, config, adat, nbp, plist, &stepWork.computeVirial);
    launchGpuKernel(kernel, config, deviceStream, timingEvent, "k_calc_nb", kernelArgs);


    bool sumUpEnergy = (stepWork.computeEnergy && c_clEnergyMemoryMultiplier > 1);
    bool sumUpShifts = (stepWork.computeVirial && c_clShiftMemoryMultiplier > 1);

    if ( sumUpEnergy || sumUpShifts )
    {
        constexpr unsigned int block_size = 64U;

        KernelLaunchConfig configSumUp;
        configSumUp.blockSize[0] = block_size;
        configSumUp.blockSize[1] = 1;
        configSumUp.blockSize[2] = 1;
        configSumUp.gridSize[0]  = sumUpShifts ? gmx::c_numShiftVectors : 1;
        configSumUp.sharedMemorySize = 0;

        const auto kernelSumUp = nbnxn_kernel_sum_up<block_size>;

        const auto kernelSumUpArgs =
                prepareGpuKernelArguments(
                    kernelSumUp,
                    configSumUp,
                    adat,
                    &gmx::c_numShiftVectors,
                    &sumUpEnergy,
                    &sumUpShifts
                );

        launchGpuKernel(
            kernelSumUp,
            configSumUp,
            deviceStream,
            nullptr,
            "nbnxn_kernel_sum_up",
            kernelSumUpArgs
        );
    }

    if (bDoTime)
    {
        timers->interaction[iloc].nb_k.closeTimingRegion(deviceStream);
    }

    if (GMX_NATIVE_WINDOWS)
    {
        /* Windows: force flushing WDDM queue */
        hipStreamQuery(deviceStream.stream());
    }
}

/*! Calculates the amount of shared memory required by the HIP kernel in use. */
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
    NBAtomDataGpu*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    Nbnxm::GpuTimers*   timers       = nb->timers;
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

    /* Don't launch the kernel if there is no work to do (not allowed with HIP) */
    if (numSciInPart <= 0)
    {
        plist->haveFreshList = false;

        return;
    }

    GpuRegionTimer* timer = nullptr;
    if (bDoTime)
    {
        timer = &(plist->haveFreshList ? timers->interaction[iloc].prune_k
                                       : timers->interaction[iloc].rollingPrune_k);
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
    int num_threads_z = c_pruneKernelJ4Concurrency;
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
                config.blockSize[0],
                config.blockSize[1],
                config.blockSize[2],
                config.gridSize[0],
                config.gridSize[1],
                numSciInPart * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster,
                plist->na_c,
                config.sharedMemorySize);
    }

    if (plist->haveFreshList)
    {
        clearDeviceBufferAsync(&plist->sci_histogram, 0, c_sciHistogramSize, deviceStream);
    }

    auto*          timingEvent  = bDoTime ? timer->fetchNextEvent() : nullptr;
    constexpr char kernelName[] = "k_pruneonly";
    const auto     kernel =
            plist->haveFreshList ? nbnxn_kernel_prune_hip<true> : nbnxn_kernel_prune_hip<false>;
    const auto kernelArgs = prepareGpuKernelArguments(kernel, config, adat, nbp, plist, &numParts, &part);
    launchGpuKernel(kernel, config, deviceStream, timingEvent, kernelName, kernelArgs);

    KernelLaunchConfig configPop;
    configPop.blockSize[0] = 128U;
    configPop.gridSize[0]  = nblock;
    configPop.sharedMemorySize = 0;

    const auto kernelPop = nbnxn_kernel_pop<128U>;

    const auto kernelPopArgs =
            prepareGpuKernelArguments(
                kernelPop,
                configPop,
                plist
            );

    launchGpuKernel(
        kernelPop,
        configPop,
        deviceStream,
        nullptr,
        "nbnxn_kernel_pop",
        kernelPopArgs
    );

    /* TODO: consider a more elegant way to track which kernel has been called
       (combined or separate 1st pass prune, rolling prune). */
    if (plist->haveFreshList)
    {
        plist->haveFreshList = false;

        size_t scan_temporary_size = (size_t)plist->nscan_temporary;
        rocprim::exclusive_scan(
            *reinterpret_cast<void**>(&plist->scan_temporary),
            scan_temporary_size,
            *reinterpret_cast<int**>(&plist->sci_histogram),
            *reinterpret_cast<int**>(&plist->sci_offset),
            0,
            c_sciHistogramSize,
            ::rocprim::plus<int>(),
            deviceStream.stream()
        );

        /*{
            std::vector<int> host_sci_histogram(plist->nsci_histogram);

            hipError_t  stat = hipMemcpy(host_sci_histogram.data(),
                                         *reinterpret_cast<int**>(&(plist->sci_histogram)),
                                         plist->nsci_histogram * sizeof(int),
                                         hipMemcpyDeviceToHost);

            std::ofstream scihistogramfile;
            scihistogramfile.open("sci_histogram.out", std::ios::app);
            scihistogramfile << "---------------------START-------------------- " << stat << std::endl;
            for(unsigned int index = 0; index < plist->nsci_histogram; index++)
            {
                scihistogramfile << index << " ; " << host_sci_histogram[index] << std::endl;
            }
            scihistogramfile << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
            scihistogramfile.close();



            std::vector<int> host_sci_count(plist->nsci_counted);

            stat = hipMemcpy(host_sci_count.data(),
                                         *reinterpret_cast<int**>(&(plist->sci_count)),
                                         plist->nsci_counted * sizeof(int),
                                         hipMemcpyDeviceToHost);

            std::ofstream scicountfile;
            scicountfile.open("sci_count.out", std::ios::app);
            scicountfile << "---------------------START-------------------- " << stat << std::endl;
            for(unsigned int index = 0; index < plist->nsci_counted; index++)
            {
                scicountfile << index << " ; " << host_sci_count[index] << std::endl;
            }
            scicountfile << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
            scicountfile.close();



            std::vector<int> host_sci_offset(plist->nsci_offset);

            stat = hipMemcpy(host_sci_offset.data(),
                             *reinterpret_cast<int**>(&(plist->sci_offset)),
                             plist->nsci_offset * sizeof(int),
                             hipMemcpyDeviceToHost);

            std::ofstream scioffsetfile;
            scioffsetfile.open("sci_offset.out", std::ios::app);
            scioffsetfile << "---------------------START-------------------- " << stat << std::endl;
            for(unsigned int index = 0; index < plist->nsci_offset; index++)
            {
                scioffsetfile << index << " ; " << host_sci_offset[index] << std::endl;
            }
            scioffsetfile << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
            scioffsetfile.close();
        }*/

        KernelLaunchConfig configSortSci;
        const unsigned int items_per_block = 256 * 16;
        configSortSci.blockSize[0] = 256;
        configSortSci.blockSize[1] = 1;
        configSortSci.blockSize[2] = 1;
        configSortSci.gridSize[0]  = (plist->nsci + items_per_block - 1) / items_per_block;
        configSortSci.sharedMemorySize = 0;

        const auto kernelSciSort = nbnxn_kernel_bucket_sci_sort<256, 16>;

        const auto kernelSciSortArgs =
                prepareGpuKernelArguments(
                    kernelSciSort,
                    configSortSci,
                    plist
                );

        launchGpuKernel(
            kernelSciSort,
            configSortSci,
            deviceStream,
            nullptr,
            "nbnxn_kernel_sci_sort",
            kernelSciSortArgs
        );

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

void hip_set_cacheconfig()
{
    hipError_t stat;

    for (int i = 0; i < c_numElecTypes; i++)
    {
        for (int j = 0; j < c_numVdwTypes; j++)
        {
            /* Default kernel 32/32 kB Shared/L1 */
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_ener_prune_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_ener_noprune_ptr[i][j]), hipFuncCachePreferEqual);
            hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_noener_prune_ptr[i][j]), hipFuncCachePreferEqual);
            stat = hipFuncSetCacheConfig(reinterpret_cast<const void*>(nb_kfunc_noener_noprune_ptr[i][j]), hipFuncCachePreferEqual);
            HIP_RET_ERR(stat, "hipFuncSetCacheConfig failed");
        }
    }
}

} // namespace Nbnxm
