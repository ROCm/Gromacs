/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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
#ifndef CUDA_ARCH_UTILS_CUH_
#define CUDA_ARCH_UTILS_CUH_

#include "gromacs/utility/basedefinitions.h"

/*! \file
 *  \brief CUDA arch dependent definitions.
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */

/* Until CC 5.2 and likely for the near future all NVIDIA architectures
   have a warp size of 32, but this could change later. If it does, the
   following constants should depend on the value of GMX_PTX_ARCH.
 */

 //#define GMX_NAVI_BUILD
 #ifdef GMX_NAVI_BUILD
 static const int warp_size      = 32;
 static const int warp_size_log2 = 5;
 #else
 static const int warp_size      = 64;
 static const int warp_size_log2 = 6;
 #endif

/*! \brief Bitmask corresponding to all threads active in a warp.
 *  NOTE that here too we assume 32-wide warps.
 */
static const unsigned long c_fullWarpMask = 0xffffffffffffffff;

/*! \brief Allow disabling CUDA textures using the GMX_DISABLE_CUDA_TEXTURES macro.
 *
 *  Only texture objects supported.
 *  Disable texture support missing in clang (all versions up to <=5.0-dev as of writing).
 *  Disable texture support on CC 7.0 and 8.0 for performance reasons (Issue #3845).
 *
 *  This option will not influence functionality. All features using textures ought
 *  to have fallback for texture-less reads (direct/LDG loads), all new code needs
 *  to provide fallback code.
 */
#    define DISABLE_CUDA_TEXTURES 1

/*! \brief True if the use of texture fetch in the CUDA kernels is disabled. */
static const bool c_disableCudaTextures = DISABLE_CUDA_TEXTURES;


/* HIP architecture technical characteristics. Needs macros because it is used
 * in the __launch_bounds__ function qualifiers and might need it in preprocessor
 * conditionals.
 *
 */
#    define GMX_CUDA_MAX_BLOCKS_PER_MP 0
#    define GMX_CUDA_MAX_THREADS_PER_MP 0

// Macro defined for clang CUDA device compilation in the presence of debug symbols
// used to work around codegen bug that breaks some kernels when assertions are on
// at -O1 and higher (tested with clang 6-8).
#    define CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE

#define HIP_PI_F 3.141592654f

#endif /* CUDA_ARCH_UTILS_CUH_ */
