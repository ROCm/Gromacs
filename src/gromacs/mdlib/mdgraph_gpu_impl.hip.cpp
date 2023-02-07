#include "hip/hip_runtime.h"
/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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
 * \brief Defines the MD Graph class
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 *
 * \ingroup module_mdlib
 */

#include "mdgraph_gpu_impl.h"

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{

MdGpuGraph::Impl::Impl(const DeviceStreamManager& deviceStreamManager,
                       SimulationWorkload         simulationWork,
                       MPI_Comm                   mpiComm,
                       MdGraphEvenOrOddStep       evenOrOddStep,
                       gmx_wallcycle*             wcycle) :
    deviceStreamManager_(deviceStreamManager),
    launchStream_(new DeviceStream(deviceStreamManager.context(), DeviceStreamPriority::Normal, false)),
    launchStreamAlternate_(
            new DeviceStream(deviceStreamManager.context(), DeviceStreamPriority::Normal, false)),
    havePPDomainDecomposition_(simulationWork.havePpDomainDecomposition),
    haveGpuPmeOnThisPpRank_(simulationWork.haveGpuPmeOnPpRank()),
    haveSeparatePmeRank_(simulationWork.haveSeparatePmeRank),
    mpiComm_(mpiComm),
    evenOrOddStep_(evenOrOddStep),
    wcycle_(wcycle)
{
    helperEvent_           = std::make_unique<GpuEventSynchronizer>();
    ppTaskCompletionEvent_ = std::make_unique<GpuEventSynchronizer>();

    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpiComm_);
        MPI_Comm_size(mpiComm_, &ppSize_);
        MPI_Comm_rank(mpiComm_, &ppRank_);
    }
}

MdGpuGraph::Impl::~Impl()
{
    stat_ = hipDeviceSynchronize();
    HIP_RET_ERR(stat_, "hipDeviceSynchronize during MD graph cleanup failed.");

    if (graphAllocated_)
    {
        stat_ = hipGraphDestroy(graph_);
        HIP_RET_ERR(stat_, "hipGraphDestroy during MD graph cleanup failed.");
    }

    if (graphInstanceAllocated_)
    {
        stat_ = hipGraphExecDestroy(instance_);
        HIP_RET_ERR(stat_, "hipGraphExecDestroy diring MD graph cleanup failed.");
    }
}


void MdGpuGraph::Impl::enqueueEventFromAllPpRanksToRank0Stream(GpuEventSynchronizer* event,
                                                               const DeviceStream&   stream)
{

    for (int remotePpRank = 1; remotePpRank < ppSize_; remotePpRank++)
    {
        if (ppRank_ == remotePpRank)
        {
            // send event to rank 0
            MPI_Send(&event,
                     sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                     MPI_BYTE,
                     0,
                     0,
                     mpiComm_);
        }
        else if (ppRank_ == 0)
        {
            // rank 0 enqueues recieved event
            GpuEventSynchronizer* eventToEnqueue;
            MPI_Recv(&eventToEnqueue,
                     sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                     MPI_BYTE,
                     remotePpRank,
                     0,
                     mpiComm_,
                     MPI_STATUS_IGNORE);
            eventToEnqueue->enqueueWaitEvent(stream);
        }
    }

    if (ppRank_ == 0)
    {
        // rank 0 also enqueues its local event
        event->enqueueWaitEvent(stream);
    }
}

void MdGpuGraph::Impl::enqueueRank0EventToAllPpStreams(GpuEventSynchronizer* event, const DeviceStream& stream)
{
    if (havePPDomainDecomposition_)
    {
        // NOLINTNEXTLINE(bugprone-sizeof-expression)
        MPI_Bcast(&event, sizeof(GpuEventSynchronizer*), MPI_BYTE, 0, mpiComm_);
    }
    event->enqueueWaitEvent(stream);
}

void MdGpuGraph::Impl::reset()
{
    graphCreated_             = false;
    useGraphThisStep_         = false;
    graphIsCapturingThisStep_ = false;
    graphState_               = GraphState::Invalid;
}

void MdGpuGraph::Impl::disableForDomainIfAnyPpRankHasCpuForces(bool disableGraphAcrossAllPpRanks)
{
    disableGraphAcrossAllPpRanks_ = disableGraphAcrossAllPpRanks;
    if (havePPDomainDecomposition_)
    {
        // If disabled on any domain, disable on all domains
        MPI_Allreduce(&disableGraphAcrossAllPpRanks,
                      &disableGraphAcrossAllPpRanks_,
                      sizeof(bool),
                      MPI_BYTE,
                      MPI_SUM,
                      mpiComm_);
    }
}

bool MdGpuGraph::Impl::captureThisStep(bool canUseGraphThisStep)
{
    useGraphThisStep_         = canUseGraphThisStep && !disableGraphAcrossAllPpRanks_;
    graphIsCapturingThisStep_ = useGraphThisStep_ && !graphCreated_;
    return graphIsCapturingThisStep_;
}

void MdGpuGraph::Impl::setUsedGraphLastStep(bool usedGraphLastStep)
{
    usedGraphLastStep_ = usedGraphLastStep;
}

void MdGpuGraph::Impl::startRecord(GpuEventSynchronizer* xReadyOnDeviceEvent)
{

    GMX_ASSERT(useGraphThisStep_,
               "startRecord should not have been called if graph is not in use this step");
    GMX_ASSERT(graphIsCapturingThisStep_,
               "startRecord should not have been called if graph is not capturing this step");
    GMX_ASSERT(graphState_ == GraphState::Invalid,
               "Graph should be in an invalid state before recording");

    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphWaitBeforeCapture);
    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpiComm_);
    }
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphWaitBeforeCapture);

    wallcycle_start(wcycle_, WallCycleCounter::MdGpuGraph);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphCapture);


    if (useGraphThisStep_ && !usedGraphLastStep_)
    {
        // Ensure NB local stream on Rank 0 (which will be used for graph capture and/or launch)
        // waits for coordinates to be ready on all ranks
        enqueueEventFromAllPpRanksToRank0Stream(
                xReadyOnDeviceEvent, deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
    }

    graphCreated_ = true;

    // Begin stream capture on PP rank 0 only. We use a single graph across all ranks.
    if (ppRank_ == 0)
    {
        stat_ = hipStreamBeginCapture(
                deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal).stream(),
                hipStreamCaptureModeGlobal);
        HIP_RET_ERR(stat_, "hipStreamBeginCapture in MD graph definition initialization failed.");
    }

    // Start artificial fork of rank>0 PP tasks from rank 0 PP
    // task, to incorporate all PP tasks into graph. This extra
    // GPU-side sync is only required to define the graph, and its
    // execution will be overlapped with completion of the graph
    // on the previous step.
    if (havePPDomainDecomposition_)
    {
        // Fork remote NB local streams from Rank 0 NB local stream
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        enqueueRank0EventToAllPpStreams(
                helperEvent_.get(), deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        // The synchronization below should not be needed, see #4674
        MPI_Barrier(mpiComm_);

        // Fork NB non-local stream from NB local stream on each rank
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        helperEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedNonLocal));
    }

    // Artificial fork of rank>0 PP tasks from rank 0 PP task has
    // now completed. Wait for PP task of graph in previous step
    // to complete its work, so that the same PP task commence its
    // work in the graph on this step
    alternateStepPpTaskCompletionEvent_->enqueueExternalWaitEventWhileCapturingGraph(
            deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

    // Fork update stream from NB local stream on each rank
    helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
    helperEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));

    // Re-mark xReadyOnDeviceEvent to allow full isolation within graph capture
    xReadyOnDeviceEvent->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));
    graphState_ = GraphState::Recording;
};


void MdGpuGraph::Impl::endRecord()
{

    GMX_ASSERT(useGraphThisStep_,
               "endRecord should not have been called if graph is not in use this step");
    GMX_ASSERT(graphIsCapturingThisStep_,
               "endRecord should not have been called if graph is not capturing this step");
    GMX_ASSERT(graphState_ == GraphState::Recording,
               "Graph should be in a recording state before recording is ended");

    if (haveGpuPmeOnThisPpRank_)
    {
        // Join PME stream to NB local stream on each rank
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::Pme));
        helperEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
    }

    // Join update stream to NB local stream on each rank
    helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));
    helperEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

    // Signal to PP task in graph of the next step that its work can commence.
    ppTaskCompletionEvent_->markExternalEventWhileCapturingGraph(
            deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

    // Start artificial join of rank>0 PP tasks to rank 0 PP
    // task, to incorporate all PP tasks into graph. This extra
    // GPU-side sync is only required to define the graph, and its
    // execution will be overlapped with the start of the graph
    // on the next step.
    if (havePPDomainDecomposition_)
    {
        // Join NB non-local stream to NB local stream on each rank
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedNonLocal));
        helperEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

        // Join remote NB local streams to Rank 0 NB local stream
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        enqueueEventFromAllPpRanksToRank0Stream(
                helperEvent_.get(), deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
    }


    // Artificial join of rank>0 PP tasks to rank 0 PP task has
    // now completed, such rank 0 PP task can end stream capture.
    if (ppRank_ == 0)
    {
        if (graphAllocated_)
        {
            stat_ = hipGraphDestroy(graph_);
            HIP_RET_ERR(stat_, "hipGraphDestroy in MD graph definition finalization failed.");
        }
        stat_ = hipStreamEndCapture(
                deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal).stream(), &graph_);
        HIP_RET_ERR(stat_, "hipStreamEndCapture in MD graph definition finalization failed.");
        graphAllocated_ = true;
    }

    graphState_ = GraphState::Recorded;

    // Sync all tasks before closing timing region, since the graph capture should be treated as a collective operation for timing purposes.
    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpiComm_);
    }
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphCapture);
    wallcycle_stop(wcycle_, WallCycleCounter::MdGpuGraph);
};

void MdGpuGraph::Impl::createExecutableGraph(bool forceGraphReinstantiation)
{

    GMX_ASSERT(
            useGraphThisStep_,
            "createExecutableGraph should not have been called if graph is not in use this step");
    GMX_ASSERT(graphIsCapturingThisStep_,
               "createExecutableGraph should not have been called if graph is not capturing this "
               "step");
    GMX_ASSERT(graphState_ == GraphState::Recorded,
               "Graph should be in a recorded state before instantiation");

    wallcycle_start(wcycle_, WallCycleCounter::MdGpuGraph);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphInstantiateOrUpdate);

    // Instantiate graph
    if (ppRank_ == 0)
    {
        // Update existing graph (which is cheaper than re-instantiation) if possible.
        // With current HIP, only single-threaded update is possible.
        // Multi-threaded update support will be available in a future HIP release.
        if (graphInstanceAllocated_ && !havePPDomainDecomposition_ && !haveSeparatePmeRank_
            && !forceGraphReinstantiation)
        {
            hipGraphNode_t           hErrorNode_out;
            hipGraphExecUpdateResult updateResult_out;
            stat_ = hipGraphExecUpdate(instance_, graph_, &hErrorNode_out, &updateResult_out);
            HIP_RET_ERR(stat_, "hipGraphExecUpdate in MD graph definition finalization failed.");
        }
        else
        {
            if (graphInstanceAllocated_)
            {
                stat_ = hipGraphExecDestroy(instance_);
                HIP_RET_ERR(stat_,
                           "hipGraphExecDestroy in MD graph definition finalization failed.");
            }
            stat_ = hipGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
            HIP_RET_ERR(stat_, "hipGraphInstantiate in MD graph definition finalization failed.");
            graphInstanceAllocated_ = true;
        }
    }

    graphState_ = GraphState::Instantiated;

    // Sync all tasks before closing timing region, since the graph instantiate or update should be treated as a collective operation for timing purposes.
    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpiComm_);
    }
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphInstantiateOrUpdate);
    wallcycle_stop(wcycle_, WallCycleCounter::MdGpuGraph);
};

void MdGpuGraph::Impl::launchGraphMdStep(GpuEventSynchronizer* xUpdatedOnDeviceEvent)
{

    GMX_ASSERT(useGraphThisStep_,
               "launchGraphMdStep should not have been called if graph is not in use this step");
    GMX_ASSERT(graphState_ == GraphState::Instantiated,
               "Graph should be in an instantiated state before launching");

    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphWaitBeforeLaunch);
    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpiComm_);
    }
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphWaitBeforeLaunch);

    wallcycle_start(wcycle_, WallCycleCounter::MdGpuGraph);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphLaunch);

    const DeviceStream* thisLaunchStream = launchStream_.get();

    // If we have multiple PP tasks, launch every alternate step in
    // alternate stream to allow overlap of extra artificial inter-GPU
    // fork and join operations across steps.
    if (havePPDomainDecomposition_ && (evenOrOddStep_ == MdGraphEvenOrOddStep::OddStep))
    {
        thisLaunchStream = launchStreamAlternate_.get();
    }

    // If graph was not used in the previous step, make sure the graph launch stream (on this step)
    // waits on all GPU streams, across all GPUs, from the previous step. First sync locally on each
    // GPU to the local launchStream, and then sync each local launchStream to the main launchStream
    // on rank 0 (noting that only rank 0 uses its launchStream to actually launch the full graph).
    if (!usedGraphLastStep_)
    {

        // Sync update and constraints
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));
        helperEvent_->enqueueWaitEvent(*thisLaunchStream);

        // Sync NB local and non-local (to ensure no race condition with pruning)
        helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        helperEvent_->enqueueWaitEvent(*thisLaunchStream);
        if (havePPDomainDecomposition_)
        {
            helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedNonLocal));
            helperEvent_->enqueueWaitEvent(*thisLaunchStream);
        }

        // If PME on same rank, sync PME (to ensure no race condition with clearing)
        // Note that separate rank PME has implicit sync, including clearing.
        if (haveGpuPmeOnThisPpRank_)
        {
            helperEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::Pme));
            helperEvent_->enqueueWaitEvent(*thisLaunchStream);
        }

        // Sync remote GPUs to main rank 0 GPU which will launch graph
        helperEvent_->markEvent(*thisLaunchStream);
        enqueueEventFromAllPpRanksToRank0Stream(helperEvent_.get(), *thisLaunchStream);
    }

    if (ppRank_ == 0)
    {
        stat_ = hipGraphLaunch(instance_, thisLaunchStream->stream());
        HIP_RET_ERR(stat_, "hipGraphLaunch in MD graph definition finalization failed.");
        helperEvent_->markEvent(*thisLaunchStream);
    }

    // ensure that "xUpdatedOnDeviceEvent" is correctly marked on all PP tasks.
    // TODO: This is actually only required on steps that don't have graph usage in
    // their following step, but it is harmless to do it on all steps for the time being.
    enqueueRank0EventToAllPpStreams(helperEvent_.get(), *thisLaunchStream);
    xUpdatedOnDeviceEvent->markEvent(*thisLaunchStream);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphLaunch);
    wallcycle_stop(wcycle_, WallCycleCounter::MdGpuGraph);
};

void MdGpuGraph::Impl::setAlternateStepPpTaskCompletionEvent(GpuEventSynchronizer* event)
{
    alternateStepPpTaskCompletionEvent_ = event;
}

GpuEventSynchronizer* MdGpuGraph::Impl::getPpTaskCompletionEvent()
{
    return ppTaskCompletionEvent_.get();
}

MdGpuGraph::MdGpuGraph(const DeviceStreamManager& deviceStreamManager,
                       SimulationWorkload         simulationWork,
                       MPI_Comm                   mpiComm,
                       MdGraphEvenOrOddStep       evenOrOddStep,
                       gmx_wallcycle*             wcycle) :
    impl_(new Impl(deviceStreamManager, simulationWork, mpiComm, evenOrOddStep, wcycle))
{
}

MdGpuGraph::~MdGpuGraph() = default;

void MdGpuGraph::reset()
{
    impl_->reset();
}

void MdGpuGraph::disableForDomainIfAnyPpRankHasCpuForces(bool disableGraphAcrossAllPpRanks)
{
    impl_->disableForDomainIfAnyPpRankHasCpuForces(disableGraphAcrossAllPpRanks);
}

bool MdGpuGraph::captureThisStep(bool canUseGraphThisStep)
{
    return impl_->captureThisStep(canUseGraphThisStep);
}

void MdGpuGraph::setUsedGraphLastStep(bool usedGraphLastStep)
{
    impl_->setUsedGraphLastStep(usedGraphLastStep);
}

void MdGpuGraph::startRecord(GpuEventSynchronizer* xReadyOnDeviceEvent)
{
    impl_->startRecord(xReadyOnDeviceEvent);
}

void MdGpuGraph::endRecord()
{
    impl_->endRecord();
}

void MdGpuGraph::createExecutableGraph(bool forceGraphReinstantiation)
{
    impl_->createExecutableGraph(forceGraphReinstantiation);
}

void MdGpuGraph::launchGraphMdStep(GpuEventSynchronizer* xUpdatedOnDeviceEvent)
{
    impl_->launchGraphMdStep(xUpdatedOnDeviceEvent);
}

bool MdGpuGraph::useGraphThisStep() const
{
    return impl_->useGraphThisStep();
}

bool MdGpuGraph::graphIsCapturingThisStep() const
{
    return impl_->graphIsCapturingThisStep();
}

void MdGpuGraph::setAlternateStepPpTaskCompletionEvent(GpuEventSynchronizer* event)
{
    impl_->setAlternateStepPpTaskCompletionEvent(event);
}

GpuEventSynchronizer* MdGpuGraph::getPpTaskCompletionEvent()
{
    return impl_->getPpTaskCompletionEvent();
}

} // namespace gmx
