#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
# Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
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

set(GMX_GPU_HIP ON)


if(GMX_DOUBLE)
    message(FATAL_ERROR "HIP acceleration is not available in double precision")
endif()

# We need to call find_package even when we've already done the detection/setup
find_package(hcc CONFIG PATHS /opt/rocm)
find_package(hip CONFIG PATHS /opt/rocm)

set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
message(STATUS "CMAKE_HIP_LINK_EXECUTABLE: " ${CMAKE_HIP_LINK_EXECUTABLE})
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
	set(HIP_CLANG_PATH "/opt/rocm/llvm/bin" CACHE PATH "Path to which HIP clang has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
	set(HIP_CLANG_PATH "/opt/rocm/llvm/bin" CACHE PATH "Path to which HIP clang has been installed")
    endif()
endif()

set(ROCM_NOTFOUND_MESSAGE "mdrun supports native GPU acceleration on ROCM hardward). This requires the ROCM HIP API, which was not found. The typical location would be /opt/rocm. Note that CPU or GPU acceleration can be selected at runtime.  ${_msg}")
unset(_msg)

if(NOT hip_FOUND)
    # the user requested ROCM, but it wasn't found
    message(FATAL_ERROR "${ROCM_NOTFOUND_MESSAGE}")
else()
    message(STATUS "HIP is found!")
endif()

find_package(rocfft QUIET CONFIG PATHS /opt/rocm)
if(NOT rocfft_FOUND)
   message(FATAL_ERROR "rocfft is required, but it is not found in this environment")
else()
   message(STATUS "rocfft is found!")
endif()

find_package(hipfft QUIET CONFIG PATHS /opt/rocm)
if(NOT hipfft_FOUND)
   message(FATAL_ERROR "hipfft is required, but it is not found in this environment")
else()
   message(STATUS "hipfft is found!")
endif()

macro(get_hip_compiler_info COMPILER_INFO DEVICE_COMPILER_FLAGS HOST_COMPILER_FLAGS)
    find_program(HIP_CONFIG hipconfig
         PATH_SUFFIXES bin
         PATHS /opt/rocm/hip
    )
    if(HIP_CONFIG)
        execute_process(COMMAND ${HIP_CONFIG} --version
                RESULT_VARIABLE _hipcc_version_res
                OUTPUT_VARIABLE _hipcc_version_out
                ERROR_VARIABLE  _hipcc_version_err
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(${_hipcc_version_res} EQUAL 0)
            find_program(HIP_CC_COMPILER hipcc
                 PATH_SUFFIXES bin
                 PATHS /opt/rocm/hip
            )
            if(HIP_CC_COMPILER)
                SET(${COMPILER_INFO} "${HIP_CC_COMPILER} ${_hipcc_version_out}")
                SET(${DEVICE_COMPILER_FLAGS} "")
            else()
                SET(${COMPILER_INFO} "N/A")
                SET(${DEVICE_COMPILER_FLAGS} "N/A")
            endif()
        else()
            SET(${COMPILER_INFO} "N/A")
            SET(${DEVICE_COMPILER_FLAGS} "N/A")
        endif()
        SET(${HOST_COMPILER_FLAGS} "")
    endif()
endmacro()

macro(enable_multiple_rocm_compilation_units)
    message(STATUS "Enabling multiple compilation units for the ROCM non-bonded module.")
    set_property(CACHE GMX_ROCM_NB_SINGLE_COMPILATION_UNIT PROPERTY VALUE OFF)
endmacro()

option(GMX_HIP_NB_SINGLE_COMPILATION_UNIT "Whether to compile the HIP non-bonded module using a single compilation unit." OFF)
mark_as_advanced(GMX_HIP_NB_SINGLE_COMPILATION_UNIT)
