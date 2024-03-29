#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2013,2014,2015,2018,2019,2020,2021, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

# Set up the module library
add_library(fft INTERFACE)

gmx_add_libgromacs_sources(
     calcgrid.cpp
     fft.cpp
     fft5d.cpp
     parallel_3dfft.cpp
     gpu_3dfft.cpp
     )

if (GMX_FFT_FFTPACK)
    gmx_add_libgromacs_sources(
        fft_fftpack.cpp
        ${CMAKE_SOURCE_DIR}/src/external/fftpack/fftpack.cpp)
endif()
if (GMX_FFT_FFTW3 OR GMX_FFT_ARMPL_FFTW3)
    gmx_add_libgromacs_sources(fft_fftw3.cpp)
endif()
if (GMX_FFT_MKL)
    gmx_add_libgromacs_sources(fft_mkl.cpp)
endif()
if(Heffte_FOUND)
    message("Adding gpu_3dfft_hefft.cpp")
    gmx_add_libgromacs_sources(
        gpu_3dfft_heffte.cpp
        )
endif()
if (GMX_GPU_CUDA)
    gmx_add_libgromacs_sources(
        # CUDA-specific sources
        gpu_3dfft_cufft.cu
        )
    _gmx_add_files_to_property(CUDA_SOURCES
        # Must add these files so they can include cuda_runtime.h
        gpu_3dfft.cpp
        )
elseif (GMX_GPU_HIP)
    gmx_add_libgromacs_sources(
        # HIP-specific sources
        gpu_3dfft_hipfft.hip.cpp
        )
    _gmx_add_files_to_property(HIP_SOURCES
        # Must add these files so they can include hip_runtime.h
        gpu_3dfft.cpp
        )
elseif (GMX_GPU_OPENCL)
    gmx_add_libgromacs_sources(
        # OpenCL-specific sources
        gpu_3dfft_ocl.cpp
        )
elseif (GMX_GPU_SYCL)
    if (NOT GMX_SYCL_HIPSYCL AND GMX_FFT_MKL)
            gmx_add_libgromacs_sources(
            gpu_3dfft_sycl_mkl.cpp
            )
        _gmx_add_files_to_property(SYCL_SOURCES
            gpu_3dfft_sycl_mkl.cpp
            )
    endif()
    if (GMX_SYCL_HIPSYCL AND GMX_HIPSYCL_HAVE_HIP_TARGET)
            gmx_add_libgromacs_sources(
            gpu_3dfft_sycl_rocfft.cpp
            )
        _gmx_add_files_to_property(SYCL_SOURCES
            gpu_3dfft_sycl_rocfft.cpp
            )
    endif()
    gmx_add_libgromacs_sources(
        gpu_3dfft_sycl.cpp
        )
    _gmx_add_files_to_property(SYCL_SOURCES
        gpu_3dfft_sycl.cpp
        gpu_3dfft.cpp
        )
else()
    gmx_add_libgromacs_sources(
        # Stub sources for CPU-only build
        gpu_3dfft.cpp
        )
endif()

# Source files have the following private module dependencies.
target_link_libraries(fft PRIVATE
                      #                      gmxlib
                      #                      math
                      #                      mdtypes
                      #                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(fft PUBLIC
target_include_directories(fft INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(fft PUBLIC
target_link_libraries(fft INTERFACE
                      legacy_api
                      )

# TODO: when fft is an OBJECT target
#target_link_libraries(fft PUBLIC legacy_api)
#target_link_libraries(fft PRIVATE common)

# Module dependencies
# This module convey transitive dependence on these modules.
#target_link_libraries(fft PUBLIC
target_link_libraries(fft INTERFACE
                      #                      utility
                      )
# Source files have the following private module dependencies.
#target_link_libraries(fft PRIVATE tng_io)
# TODO: Explicitly link specific modules.
#target_link_libraries(fft PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
