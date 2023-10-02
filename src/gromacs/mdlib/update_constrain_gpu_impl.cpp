/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 *
 * \brief Implements update and constraints class.
 *
 * The class combines Leap-Frog integrator with LINCS and SETTLE constraints.
 *
 * \todo The computational procedures in members should be integrated to improve
 *       computational performance.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "update_constrain_gpu_impl.h"

#include <assert.h>
#include <stdio.h>

#include <cmath>

#include <algorithm>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/mdlib/leapfrog_gpu.h"
#include "gromacs/mdlib/update_constrain_gpu.h"
#include "gromacs/mdlib/update_constrain_gpu_internal.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/mtop_util.h"

static constexpr bool sc_haveGpuConstraintSupport = (GMX_GPU_CUDA || GMX_GPU_HIP || GMX_GPU_SYCL);

namespace gmx
{

void UpdateConstrainGpu::Impl::integrate(GpuEventSynchronizer*             fReadyOnDevice,
                                         const real                        dt,
                                         const real                        dttc,
                                         const bool                        updateVelocities,
                                         const bool                        computeVirial,
                                         tensor                            virial,
                                         const bool                        doTemperatureScaling,
                                         const bool                        doNoseHoover, 
                                         gmx::ArrayRef<const t_grp_tcstat> tcstat,
                                         const bool                        doParrinelloRahman,
                                         const float                       dtPressureCouple,
                                         const bool                        isPmeRank, 
                                         const matrix                      prVelocityScalingMatrix)
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    // Clearing virial matrix
    // TODO There is no point in having separate virial matrix for constraints
    clear_mat(virial);

    // Make sure that the forces are ready on device before proceeding with the update.
    fReadyOnDevice->enqueueWaitEvent(deviceStream_);

    // The integrate should save a copy of the current coordinates in d_xp_ and write updated
    // once into d_x_. The d_xp_ is only needed by constraints
    integrator_->integrate(
            d_x_, d_xp_, d_v_, realGridSize_, *d_grid_, 
            d_f_, d_reft_, d_th_, d_massQInv_, d_xi_, d_vxi_, dt,dttc, doTemperatureScaling, 
            doNoseHoover, tcstat, doParrinelloRahman, dtPressureCouple, isPmeRank,  prVelocityScalingMatrix);
    // Constraints need both coordinates before (d_x_) and after (d_xp_) update. However, after constraints
    // are applied, the d_x_ can be discarded. So we intentionally swap the d_x_ and d_xp_ here to avoid the
    // d_xp_ -> d_x_ copy after constraints. Note that the integrate saves them in the wrong order as well.
    if (sc_haveGpuConstraintSupport)
    {
        lincsGpu_->apply(d_xp_, d_x_, updateVelocities, d_v_, 1.0 / dt, computeVirial, virial, pbcAiuc_);
        settleGpu_->apply(d_xp_, d_x_, updateVelocities, d_v_, 1.0 / dt, computeVirial, virial, pbcAiuc_);
    }

    // scaledVirial -> virial (methods above returns scaled values)
    float scaleFactor = 0.5F / (dt * dt);
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            virial[i][j] = scaleFactor * virial[i][j];
        }
    }

    xUpdatedOnDeviceEvent_.markEvent(deviceStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

void UpdateConstrainGpu::Impl::scaleCoordinates(const matrix scalingMatrix)
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    ScalingMatrix mu(scalingMatrix);

    launchScaleCoordinatesKernel(numAtoms_, d_x_, mu, deviceStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

void UpdateConstrainGpu::Impl::scaleVelocities(const matrix scalingMatrix)
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    ScalingMatrix mu(scalingMatrix);

    launchScaleCoordinatesKernel(numAtoms_, d_v_, mu, deviceStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

UpdateConstrainGpu::Impl::Impl(const t_inputrec&    ir,
                               const gmx_mtop_t&    mtop,
                               const int            numTempScaleValues,
                               const DeviceContext& deviceContext,
                               const DeviceStream&  deviceStream,
                               gmx_wallcycle*       wcycle) :
    deviceContext_(deviceContext), deviceStream_(deviceStream), wcycle_(wcycle)
{
    integrator_ = std::make_unique<LeapFrogGpu>(deviceContext_, deviceStream_, numTempScaleValues);
    if (sc_haveGpuConstraintSupport)
    {
        lincsGpu_ = std::make_unique<LincsGpu>(ir.nLincsIter, ir.nProjOrder, deviceContext_, deviceStream_);
        settleGpu_ = std::make_unique<SettleGpu>(mtop, deviceContext_, deviceStream_);
    }
}

UpdateConstrainGpu::Impl::~Impl() {}

void UpdateConstrainGpu::Impl::set(DeviceBuffer<Float3>          d_x,
                                   DeviceBuffer<Float3>          d_v, 
                                   const int                     realGridSize, 
                                   DeviceBuffer<real>*           d_grid, 
                                   const DeviceBuffer<Float3>    d_f,
                                   DeviceBuffer<float>           d_reft, 
                                   DeviceBuffer<float>           d_th, 
                                   DeviceBuffer<float>           d_massQInv, 
                                   DeviceBuffer<float>           d_xi, 
                                   DeviceBuffer<float>           d_vxi,
                                   const InteractionDefinitions& idef,
                                   const t_mdatoms&              md)
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    GMX_ASSERT(d_x, "Coordinates device buffer should not be null.");
    GMX_ASSERT(d_v, "Velocities device buffer should not be null.");
    GMX_ASSERT(d_f, "Forces device buffer should not be null.");

    d_x_ = d_x;
    d_v_ = d_v;
    d_f_ = d_f;
    d_reft_ = d_reft;
    d_th_   = d_th;
    d_massQInv_ = d_massQInv;
    d_xi_   = d_xi;
    d_vxi_  = d_vxi;
    realGridSize_ = realGridSize;
    d_grid_ = d_grid;

    numAtoms_ = md.nr;

    reallocateDeviceBuffer(&d_xp_, numAtoms_, &numXp_, &numXpAlloc_, deviceContext_);

    reallocateDeviceBuffer(
            &d_inverseMasses_, numAtoms_, &numInverseMasses_, &numInverseMassesAlloc_, deviceContext_);

    // Integrator should also update something, but it does not even have a method yet
    integrator_->set(numAtoms_, md.invmass, md.cTC);
    if (sc_haveGpuConstraintSupport)
    {
        lincsGpu_->set(idef, numAtoms_, md.invmass);
        settleGpu_->set(idef);
    }
    else
    {
        GMX_ASSERT(idef.il[F_SETTLE].empty(), "SETTLE not supported");
        GMX_ASSERT(idef.il[F_CONSTR].empty(), "LINCS not supported");
    }

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

void UpdateConstrainGpu::Impl::setPbc(const PbcType pbcType, const matrix box)
{
    // TODO wallcycle
    setPbcAiuc(numPbcDimensions(pbcType), box, &pbcAiuc_);
}

GpuEventSynchronizer* UpdateConstrainGpu::Impl::xUpdatedOnDeviceEvent()
{
    return &xUpdatedOnDeviceEvent_;
}

UpdateConstrainGpu::UpdateConstrainGpu(const t_inputrec&    ir,
                                       const gmx_mtop_t&    mtop,
                                       const int            numTempScaleValues,
                                       const DeviceContext& deviceContext,
                                       const DeviceStream&  deviceStream,
                                       gmx_wallcycle*       wcycle) :
    impl_(new Impl(ir, mtop, numTempScaleValues, deviceContext, deviceStream, wcycle))
{
}

UpdateConstrainGpu::~UpdateConstrainGpu() = default;

void UpdateConstrainGpu::integrate(GpuEventSynchronizer*             fReadyOnDevice,
                                   const real                        dt,
                                   const real                        dttc,
                                   const bool                        updateVelocities,
                                   const bool                        computeVirial,
                                   tensor                            virialScaled,
                                   const bool                        doTemperatureScaling,
                                   const bool                        doNoseHoover, 
                                   gmx::ArrayRef<const t_grp_tcstat> tcstat,
                                   const bool                        doParrinelloRahman,
                                   const float                       dtPressureCouple,
                                   const bool                        isPmeRank, 
                                   const matrix                      prVelocityScalingMatrix)
{
    impl_->integrate(fReadyOnDevice,
                     dt,
                     dttc, 
                     updateVelocities,
                     computeVirial,
                     virialScaled,
                     doTemperatureScaling,
                     doNoseHoover, 
                     tcstat,
                     doParrinelloRahman,
                     dtPressureCouple,
                     isPmeRank, 
                     prVelocityScalingMatrix);
}

void UpdateConstrainGpu::scaleCoordinates(const matrix scalingMatrix)
{
    impl_->scaleCoordinates(scalingMatrix);
}

void UpdateConstrainGpu::scaleVelocities(const matrix scalingMatrix)
{
    impl_->scaleVelocities(scalingMatrix);
}

void UpdateConstrainGpu::set(DeviceBuffer<Float3>          d_x,
                             DeviceBuffer<Float3>          d_v,
                             const int                     realGridSize, 
                             DeviceBuffer<real>*           d_grid,
                             const DeviceBuffer<Float3>    d_f,
                             DeviceBuffer<float>           d_reft, 
                             DeviceBuffer<float>           d_th, 
                             DeviceBuffer<float>           d_massQInv, 
                             DeviceBuffer<float>           d_xi, 
                             DeviceBuffer<float>           d_vxi,
                             const InteractionDefinitions& idef,
                             const t_mdatoms&              md)
{
    impl_->set(d_x, d_v, realGridSize, d_grid, d_f, d_reft, d_th, d_massQInv, d_xi, d_vxi, idef, md);
}

void UpdateConstrainGpu::setPbc(const PbcType pbcType, const matrix box)
{
    impl_->setPbc(pbcType, box);
}

GpuEventSynchronizer* UpdateConstrainGpu::xUpdatedOnDeviceEvent()
{
    return impl_->xUpdatedOnDeviceEvent();
}

bool UpdateConstrainGpu::isNumCoupledConstraintsSupported(const gmx_mtop_t& mtop)
{
    return LincsGpu::isNumCoupledConstraintsSupported(mtop);
}

bool UpdateConstrainGpu::areConstraintsSupported()
{
    return sc_haveGpuConstraintSupport;
}

} // namespace gmx
