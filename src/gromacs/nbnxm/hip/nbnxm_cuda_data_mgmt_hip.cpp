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
 *  \brief Define CUDA implementation of nbnxn_gpu_data_mgmt.h
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */
#include "gmxpre.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// TODO We would like to move this down, but the way NbnxmGpu
//      is currently declared means this has to be before gpu_types.h
#include "nbnxm_cuda_types.h"

// TODO Remove this comment when the above order issue is resolved
#include "gromacs/gpu_utils/cudautils_hip.h"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gpueventsynchronizer_hip.h"
#include "gromacs/gpu_utils/pmalloc_cuda.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_data_mgmt.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "nbnxm_cuda.h"

namespace Nbnxm
{

/* This is a heuristically determined parameter for the Kepler
 * and Maxwell architectures for the minimum size of ci lists by multiplying
 * this constant with the # of multiprocessors on the current device.
 * Since the maximum number of blocks per multiprocessor is 16, the ideal
 * count for small systems is 32 or 48 blocks per multiprocessor. Because
 * there is a bit of fluctuations in the generated block counts, we use
 * a target of 44 instead of the ideal value of 48.
 */
static unsigned int gpu_min_ci_balanced_factor = 44;

/* Fw. decl. */
static void nbnxn_cuda_clear_e_fshift(NbnxmGpu* nb);

/*! Initializes the atomdata structure first time, it only gets filled at
    pair-search. */
static void init_atomdata_first(cu_atomdata_t* ad, int ntypes, const DeviceContext& deviceContext)
{
    ad->ntypes = ntypes;
    allocateDeviceBuffer(&ad->shift_vec, SHIFTS, deviceContext);
    ad->bShiftVecUploaded = false;

    allocateDeviceBuffer(&ad->fshift, c_clShiftMemorySize * SHIFTS, deviceContext);
    allocateDeviceBuffer(&ad->e_lj, c_clEnergyMemorySize, deviceContext);
    allocateDeviceBuffer(&ad->e_el, c_clEnergyMemorySize, deviceContext);

    /* initialize to nullptr poiters to data that is not allocated here and will
       need reallocation in nbnxn_cuda_init_atomdata */
    ad->xq = nullptr;
    ad->f  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    ad->natoms = -1;
    ad->nalloc = -1;
}

/*! Initializes the nonbonded parameter data structure. */
static void init_nbparam(NBParamGpu*                     nbp,
                         const interaction_const_t*      ic,
                         const PairlistParams&           listParams,
                         const nbnxn_atomdata_t::Params& nbatParams,
                         const DeviceContext&            deviceContext)
{
    int ntypes;

    ntypes = nbatParams.numTypes;

    set_cutoff_parameters(nbp, ic, listParams);

    /* The kernel code supports LJ combination rules (geometric and LB) for
     * all kernel types, but we only generate useful combination rule kernels.
     * We currently only use LJ combination rule (geometric and LB) kernels
     * for plain cut-off LJ. On Maxwell the force only kernels speed up 15%
     * with PME and 20% with RF, the other kernels speed up about half as much.
     * For LJ force-switch the geometric rule would give 7% speed-up, but this
     * combination is rarely used. LJ force-switch with LB rule is more common,
     * but gives only 1% speed-up.
     */
    if (ic->vdwtype == evdwCUT)
    {
        switch (ic->vdw_modifier)
        {
            case eintmodNONE:
            case eintmodPOTSHIFT:
                switch (nbatParams.comb_rule)
                {
                    case ljcrNONE: nbp->vdwtype = evdwTypeCUT; break;
                    case ljcrGEOM: nbp->vdwtype = evdwTypeCUTCOMBGEOM; break;
                    case ljcrLB: nbp->vdwtype = evdwTypeCUTCOMBLB; break;
                    default:
                        gmx_incons(
                                "The requested LJ combination rule is not implemented in the CUDA "
                                "GPU accelerated kernels!");
                }
                break;
            case eintmodFORCESWITCH: nbp->vdwtype = evdwTypeFSWITCH; break;
            case eintmodPOTSWITCH: nbp->vdwtype = evdwTypePSWITCH; break;
            default:
                gmx_incons(
                        "The requested VdW interaction modifier is not implemented in the CUDA GPU "
                        "accelerated kernels!");
        }
    }
    else if (ic->vdwtype == evdwPME)
    {
        if (ic->ljpme_comb_rule == ljcrGEOM)
        {
            assert(nbatParams.comb_rule == ljcrGEOM);
            nbp->vdwtype = evdwTypeEWALDGEOM;
        }
        else
        {
            assert(nbatParams.comb_rule == ljcrLB);
            nbp->vdwtype = evdwTypeEWALDLB;
        }
    }
    else
    {
        gmx_incons(
                "The requested VdW type is not implemented in the CUDA GPU accelerated kernels!");
    }

    if (ic->eeltype == eelCUT)
    {
        nbp->eeltype = eelTypeCUT;
    }
    else if (EEL_RF(ic->eeltype))
    {
        nbp->eeltype = eelTypeRF;
    }
    else if ((EEL_PME(ic->eeltype) || ic->eeltype == eelEWALD))
    {
        nbp->eeltype = nbnxn_gpu_pick_ewald_kernel_type(*ic, deviceContext.deviceInfo());
    }
    else
    {
        /* Shouldn't happen, as this is checked when choosing Verlet-scheme */
        gmx_incons(
                "The requested electrostatics type is not implemented in the CUDA GPU accelerated "
                "kernels!");
    }

    /* generate table for PME */
    nbp->coulomb_tab = nullptr;
    if (nbp->eeltype == eelTypeEWALD_TAB || nbp->eeltype == eelTypeEWALD_TAB_TWIN)
    {
        GMX_RELEASE_ASSERT(ic->coulombEwaldTables, "Need valid Coulomb Ewald correction tables");
        init_ewald_coulomb_force_table(*ic->coulombEwaldTables, nbp, deviceContext);
    }

    /* set up LJ parameter lookup table */
    if (!useLjCombRule(nbp->vdwtype))
    {
        initParamLookupTable(&nbp->nbfp, &nbp->nbfp_texobj, nbatParams.nbfp.data(),
                             2 * ntypes * ntypes, deviceContext);
    }

    /* set up LJ-PME parameter lookup table */
    if (ic->vdwtype == evdwPME)
    {
        initParamLookupTable(&nbp->nbfp_comb, &nbp->nbfp_comb_texobj, nbatParams.nbfp_comb.data(),
                             2 * ntypes, deviceContext);
    }
}

/*! Initializes simulation constant data. */
static void cuda_init_const(NbnxmGpu*                       nb,
                            const interaction_const_t*      ic,
                            const PairlistParams&           listParams,
                            const nbnxn_atomdata_t::Params& nbatParams)
{
    init_atomdata_first(nb->atdat, nbatParams.numTypes, *nb->deviceContext_);
    init_nbparam(nb->nbparam, ic, listParams, nbatParams, *nb->deviceContext_);

    /* clear energy and shift force outputs */
    nbnxn_cuda_clear_e_fshift(nb);
}

NbnxmGpu* gpu_init(const gmx::DeviceStreamManager& deviceStreamManager,
                   const interaction_const_t*      ic,
                   const PairlistParams&           listParams,
                   const nbnxn_atomdata_t*         nbat,
                   bool                            bLocalAndNonlocal)
{
    hipError_t stat;

    auto nb            = new NbnxmGpu();
    nb->deviceContext_ = &deviceStreamManager.context();
    snew(nb->atdat, 1);
    snew(nb->nbparam, 1);
    snew(nb->plist[InteractionLocality::Local], 1);
    if (bLocalAndNonlocal)
    {
        snew(nb->plist[InteractionLocality::NonLocal], 1);
    }

    nb->bUseTwoStreams = bLocalAndNonlocal;

    nb->timers = new cu_timers_t();
    snew(nb->timings, 1);

    /* init nbst */
    pmalloc((void**)&nb->nbst.e_lj, sizeof(*nb->nbst.e_lj));
    pmalloc((void**)&nb->nbst.e_el, sizeof(*nb->nbst.e_el));
    pmalloc((void**)&nb->nbst.fshift, SHIFTS * sizeof(*nb->nbst.fshift));

    init_plist(nb->plist[InteractionLocality::Local]);

    /* local/non-local GPU streams */
    GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedLocal),
                       "Local non-bonded stream should be initialized to use GPU for non-bonded.");
    nb->deviceStreams[InteractionLocality::Local] =
            &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal);
    if (nb->bUseTwoStreams)
    {
        init_plist(nb->plist[InteractionLocality::NonLocal]);

        /* Note that the device we're running on does not have to support
         * priorities, because we are querying the priority range which in this
         * case will be a single value.
         */
        GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedNonLocal),
                           "Non-local non-bonded stream should be initialized to use GPU for "
                           "non-bonded with domain decomposition.");
        nb->deviceStreams[InteractionLocality::NonLocal] =
                &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedNonLocal);
        ;
    }

    /* init events for sychronization (timing disabled for performance reasons!) */
    stat = hipEventCreateWithFlags(&nb->nonlocal_done, hipEventDisableTiming);
    CU_RET_ERR(stat, "hipEventCreate on nonlocal_done failed");
    stat = hipEventCreateWithFlags(&nb->misc_ops_and_local_H2D_done, hipEventDisableTiming);
    CU_RET_ERR(stat, "hipEventCreate on misc_ops_and_local_H2D_done failed");

    nb->xNonLocalCopyD2HDone = new GpuEventSynchronizer();

    /* WARNING: CUDA timings are incorrect with multiple streams.
     *          This is the main reason why they are disabled by default.
     */
    // TODO: Consider turning on by default when we can detect nr of streams.
    nb->bDoTime = (getenv("GMX_ENABLE_GPU_TIMING") != nullptr);

    if (nb->bDoTime)
    {
        init_timings(nb->timings);
    }

    /* set the kernel type for the current GPU */
    /* pick L1 cache configuration */
    //cuda_set_cacheconfig();

    cuda_init_const(nb, ic, listParams, nbat->params());

    nb->atomIndicesSize       = 0;
    nb->atomIndicesSize_alloc = 0;
    nb->ncxy_na               = 0;
    nb->ncxy_na_alloc         = 0;
    nb->ncxy_ind              = 0;
    nb->ncxy_ind_alloc        = 0;

    if (debug)
    {
        fprintf(debug, "Initialized CUDA data structures.\n");
    }

    return nb;
}

void gpu_upload_shiftvec(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom)
{
    cu_atomdata_t*      adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    /* only if we have a dynamic box */
    if (nbatom->bDynamicBox || !adat->bShiftVecUploaded)
    {
        static_assert(sizeof(adat->shift_vec[0]) == sizeof(nbatom->shift_vec[0]),
                      "Sizes of host- and device-side shift vectors should be the same.");
        copyToDeviceBuffer(&adat->shift_vec, reinterpret_cast<const float3*>(nbatom->shift_vec.data()),
                           0, SHIFTS, localStream, GpuApiCallBehavior::Async, nullptr);
        adat->bShiftVecUploaded = true;
    }
}

/*! Clears the first natoms_clear elements of the GPU nonbonded force output array. */
static void nbnxn_cuda_clear_f(NbnxmGpu* nb, int natoms_clear)
{
    cu_atomdata_t*      adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];
    clearDeviceBufferAsync(&adat->f, 0, natoms_clear, localStream);
}

/*! Clears nonbonded shift force output array and energy outputs on the GPU. */
static void nbnxn_cuda_clear_e_fshift(NbnxmGpu* nb)
{
    cu_atomdata_t*      adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    clearDeviceBufferAsync(&adat->fshift, 0, c_clShiftMemorySize * SHIFTS, localStream);
    clearDeviceBufferAsync(&adat->e_lj, 0, c_clEnergyMemorySize, localStream);
    clearDeviceBufferAsync(&adat->e_el, 0, c_clEnergyMemorySize, localStream);
}

void gpu_clear_outputs(NbnxmGpu* nb, bool computeVirial)
{
    nbnxn_cuda_clear_f(nb, nb->atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (computeVirial)
    {
        nbnxn_cuda_clear_e_fshift(nb);
    }
}

void gpu_init_atomdata(NbnxmGpu* nb, const nbnxn_atomdata_t* nbat)
{
    int                  nalloc, natoms;
    bool                 realloced;
    bool                 bDoTime       = nb->bDoTime;
    cu_timers_t*         timers        = nb->timers;
    cu_atomdata_t*       d_atdat       = nb->atdat;
    const DeviceContext& deviceContext = *nb->deviceContext_;
    const DeviceStream&  localStream   = *nb->deviceStreams[InteractionLocality::Local];

    natoms    = nbat->numAtoms();
    realloced = false;

    if (bDoTime)
    {
        /* time async copy */
        timers->atdat.openTimingRegion(localStream);
    }

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > d_atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (d_atdat->nalloc != -1)
        {
            freeDeviceBuffer(&d_atdat->f);
            freeDeviceBuffer(&d_atdat->xq);
            freeDeviceBuffer(&d_atdat->atom_types);
            freeDeviceBuffer(&d_atdat->lj_comb);
        }

        allocateDeviceBuffer(&d_atdat->f, nalloc, deviceContext);
        allocateDeviceBuffer(&d_atdat->xq, nalloc, deviceContext);
        if (useLjCombRule(nb->nbparam->vdwtype))
        {
            allocateDeviceBuffer(&d_atdat->lj_comb, nalloc, deviceContext);
        }
        else
        {
            allocateDeviceBuffer(&d_atdat->atom_types, nalloc, deviceContext);
        }

        d_atdat->nalloc = nalloc;
        realloced       = true;
    }

    d_atdat->natoms       = natoms;
    d_atdat->natoms_local = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        nbnxn_cuda_clear_f(nb, nalloc);
    }

    if (useLjCombRule(nb->nbparam->vdwtype))
    {
        static_assert(sizeof(d_atdat->lj_comb[0]) == sizeof(float2),
                      "Size of the LJ parameters element should be equal to the size of float2.");
        copyToDeviceBuffer(&d_atdat->lj_comb,
                           reinterpret_cast<const float2*>(nbat->params().lj_comb.data()), 0,
                           natoms, localStream, GpuApiCallBehavior::Async, nullptr);
    }
    else
    {
        static_assert(sizeof(d_atdat->atom_types[0]) == sizeof(nbat->params().type[0]),
                      "Sizes of host- and device-side atom types should be the same.");
        copyToDeviceBuffer(&d_atdat->atom_types, nbat->params().type.data(), 0, natoms, localStream,
                           GpuApiCallBehavior::Async, nullptr);
    }

    if (bDoTime)
    {
        timers->atdat.closeTimingRegion(localStream);
    }
}

void gpu_free(NbnxmGpu* nb)
{
    hipError_t    stat;
    cu_atomdata_t* atdat;
    NBParamGpu*    nbparam;

    if (nb == nullptr)
    {
        return;
    }

    atdat   = nb->atdat;
    nbparam = nb->nbparam;

    if ((!nbparam->coulomb_tab)
        && (nbparam->eeltype == eelTypeEWALD_TAB || nbparam->eeltype == eelTypeEWALD_TAB_TWIN))
    {
        destroyParamLookupTable(&nbparam->coulomb_tab, nbparam->coulomb_tab_texobj);
    }

    stat = hipEventDestroy(nb->nonlocal_done);
    CU_RET_ERR(stat, "hipEventDestroy failed on timers->nonlocal_done");
    stat = hipEventDestroy(nb->misc_ops_and_local_H2D_done);
    CU_RET_ERR(stat, "hipEventDestroy failed on timers->misc_ops_and_local_H2D_done");

    delete nb->timers;

    if (!useLjCombRule(nb->nbparam->vdwtype))
    {
        destroyParamLookupTable(&nbparam->nbfp, nbparam->nbfp_texobj);
    }

    if (nbparam->vdwtype == evdwTypeEWALDGEOM || nbparam->vdwtype == evdwTypeEWALDLB)
    {
        destroyParamLookupTable(&nbparam->nbfp_comb, nbparam->nbfp_comb_texobj);
    }

    freeDeviceBuffer(&atdat->shift_vec);
    freeDeviceBuffer(&atdat->fshift);

    freeDeviceBuffer(&atdat->e_lj);
    freeDeviceBuffer(&atdat->e_el);

    freeDeviceBuffer(&atdat->f);
    freeDeviceBuffer(&atdat->xq);
    freeDeviceBuffer(&atdat->atom_types);
    freeDeviceBuffer(&atdat->lj_comb);

    /* Free plist */
    auto* plist = nb->plist[InteractionLocality::Local];
    freeDeviceBuffer(&plist->sci);
    freeDeviceBuffer(&plist->cj4);
    freeDeviceBuffer(&plist->imask);
    freeDeviceBuffer(&plist->excl);
    sfree(plist);
    if (nb->bUseTwoStreams)
    {
        auto* plist_nl = nb->plist[InteractionLocality::NonLocal];
        freeDeviceBuffer(&plist_nl->sci);
        freeDeviceBuffer(&plist_nl->cj4);
        freeDeviceBuffer(&plist_nl->imask);
        freeDeviceBuffer(&plist_nl->excl);
        sfree(plist_nl);
    }

    /* Free nbst */
    pfree(nb->nbst.e_lj);
    nb->nbst.e_lj = nullptr;

    pfree(nb->nbst.e_el);
    nb->nbst.e_el = nullptr;

    pfree(nb->nbst.fshift);
    nb->nbst.fshift = nullptr;

    sfree(atdat);
    sfree(nbparam);
    sfree(nb->timings);
    delete nb;

    if (debug)
    {
        fprintf(debug, "Cleaned up CUDA data structures.\n");
    }
}

int gpu_min_ci_balanced(NbnxmGpu* nb)
{
    return nb != nullptr ? gpu_min_ci_balanced_factor * nb->deviceContext_->deviceInfo().prop.multiProcessorCount
                         : 0;
}

void* gpu_get_xq(NbnxmGpu* nb)
{
    assert(nb);

    return static_cast<void*>(nb->atdat->xq);
}

DeviceBuffer<gmx::RVec> gpu_get_f(NbnxmGpu* nb)
{
    assert(nb);

    return reinterpret_cast<DeviceBuffer<gmx::RVec>>(nb->atdat->f);
}

DeviceBuffer<gmx::RVec> gpu_get_fshift(NbnxmGpu* nb)
{
    assert(nb);

    return reinterpret_cast<DeviceBuffer<gmx::RVec>>(nb->atdat->fshift);
}

/* Initialization for X buffer operations on GPU. */
/* TODO  Remove explicit pinning from host arrays from here and manage in a more natural way*/
void nbnxn_gpu_init_x_to_nbat_x(const Nbnxm::GridSet& gridSet, NbnxmGpu* gpu_nbv)
{
    const DeviceStream& deviceStream  = *gpu_nbv->deviceStreams[InteractionLocality::Local];
    bool                bDoTime       = gpu_nbv->bDoTime;
    const int           maxNumColumns = gridSet.numColumnsMax();

    reallocateDeviceBuffer(&gpu_nbv->cxy_na, maxNumColumns * gridSet.grids().size(),
                           &gpu_nbv->ncxy_na, &gpu_nbv->ncxy_na_alloc, *gpu_nbv->deviceContext_);
    reallocateDeviceBuffer(&gpu_nbv->cxy_ind, maxNumColumns * gridSet.grids().size(),
                           &gpu_nbv->ncxy_ind, &gpu_nbv->ncxy_ind_alloc, *gpu_nbv->deviceContext_);

    for (unsigned int g = 0; g < gridSet.grids().size(); g++)
    {

        const Nbnxm::Grid& grid = gridSet.grids()[g];

        const int  numColumns      = grid.numColumns();
        const int* atomIndices     = gridSet.atomIndices().data();
        const int  atomIndicesSize = gridSet.atomIndices().size();
        const int* cxy_na          = grid.cxy_na().data();
        const int* cxy_ind         = grid.cxy_ind().data();

        reallocateDeviceBuffer(&gpu_nbv->atomIndices, atomIndicesSize, &gpu_nbv->atomIndicesSize,
                               &gpu_nbv->atomIndicesSize_alloc, *gpu_nbv->deviceContext_);

        if (atomIndicesSize > 0)
        {

            if (bDoTime)
            {
                gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d.openTimingRegion(deviceStream);
            }

            copyToDeviceBuffer(&gpu_nbv->atomIndices, atomIndices, 0, atomIndicesSize, deviceStream,
                               GpuApiCallBehavior::Async, nullptr);

            if (bDoTime)
            {
                gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d.closeTimingRegion(deviceStream);
            }
        }

        if (numColumns > 0)
        {
            if (bDoTime)
            {
                gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d.openTimingRegion(deviceStream);
            }

            int* destPtr = &gpu_nbv->cxy_na[maxNumColumns * g];
            copyToDeviceBuffer(&destPtr, cxy_na, 0, numColumns, deviceStream,
                               GpuApiCallBehavior::Async, nullptr);

            if (bDoTime)
            {
                gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d.closeTimingRegion(deviceStream);
            }

            if (bDoTime)
            {
                gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d.openTimingRegion(deviceStream);
            }

            destPtr = &gpu_nbv->cxy_ind[maxNumColumns * g];
            copyToDeviceBuffer(&destPtr, cxy_ind, 0, numColumns, deviceStream,
                               GpuApiCallBehavior::Async, nullptr);

            if (bDoTime)
            {
                gpu_nbv->timers->xf[AtomLocality::Local].nb_h2d.closeTimingRegion(deviceStream);
            }
        }
    }

    // The above data is transferred on the local stream but is a
    // dependency of the nonlocal stream (specifically the nonlocal X
    // buf ops kernel).  We therefore set a dependency to ensure
    // that the nonlocal stream waits on the local stream here.
    // This call records an event in the local stream:
    nbnxnInsertNonlocalGpuDependency(gpu_nbv, Nbnxm::InteractionLocality::Local);
    // ...and this call instructs the nonlocal stream to wait on that event:
    nbnxnInsertNonlocalGpuDependency(gpu_nbv, Nbnxm::InteractionLocality::NonLocal);

    return;
}

} // namespace Nbnxm
