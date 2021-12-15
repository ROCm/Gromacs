#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2017,2018,2019,2020,2021, by the GROMACS development team, led by
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

function (gmx_test_clang_hip_support)

    if ((NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR
        (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6))
        message(FATAL_ERROR "clang 6 or later required with GMX_CLANG_HIP=ON!")
    endif()

    # NOTE: we'd ideally like to use a compile check here, but the link-stage
    # fails as the clang invocation generated seems to not handle well some
    # (GPU code) in the object file generated during compilation.
    # SET(CMAKE_REQUIRED_FLAGS ${FLAGS})
    # SET(CMAKE_REQUIRED_LIBRARIES ${LIBS})
    # CHECK_CXX_SOURCE_COMPILES("int main() { int c; hipGetDeviceCount(&c); return 0; }" _CLANG_HIP_COMPILES)
endfunction ()

if (GMX_HIP_TARGET_COMPUTE)
    message(WARNING "Values passed in GMX_HIP_TARGET_COMPUTE will be ignored; clang will by default include PTX in the binary.")
endif()

if (HIP_VERSION VERSION_GREATER 10.1)
    # At the time of writing, the latest versions are Clang 11 and HIP 11.2.
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.0)
        # We don't know about the future Clang versions, but so far Clang 12 docs state that only HIP versions 7.0-10.1 are supported.
        set(_support_status "likely incompatible")
    else()
        if (HIP_VERSION VERSION_GREATER 11.2)
            # No idea about future HIP versions.
            set(_support_status "officially incompatible")
        else()
            # Our experience and multiple reports on the internet indicate that it works just fine.
            set(_support_status "officially incompatible (but generally working)")
        endif()
    endif()
    message(NOTICE "Using ${_support_status} version of HIP with Clang.")
    message(NOTICE "If Clang fails to recognize HIP version, consider creating doing "
      "`echo \"HIP Version ${HIP_VERSION}\" | sudo tee \"${HIP_TOOLKIT_ROOT_DIR}/version.txt\"`")
    list(APPEND _HIP_CLANG_FLAGS "-Wno-unknown-hip-version")

endif()

if (GMX_HIP_TARGET_SM)
    set(_HIP_CLANG_GENCODE_FLAGS)
    set(_target_sm_list ${GMX_HIP_TARGET_SM})
    foreach(_target ${_target_sm_list})
        list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_${_target}")
    endforeach()
else()
    if (HIP_VERSION VERSION_LESS 11.0)
        list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_30")
    endif()
    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_35")
    # clang 6.0 + HIP 9.0 seems to have issues generating code for sm_37
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0.999)
        list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_37")
    endif()
    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_50")
    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_52")
    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_60")
    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_61")
    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_70")
    if (NOT HIP_VERSION VERSION_LESS 10.0)
        list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_75")
    endif()
    # Enable this when clang (12.0 ?) properly recognizes HIP 11.0
    #if(NOT HIP_VERSION VERSION_LESS 11.0)
    #    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_80")
    #endif()
    # Enable this when clang (12.0 ?) introduces sm_86 support
    #if(NOT HIP_VERSION VERSION_LESS 11.1)
    #    list(APPEND _HIP_CLANG_GENCODE_FLAGS "--hip-gpu-arch=sm_86")
    #endif()
endif()
if (GMX_HIP_TARGET_SM)
    set_property(CACHE GMX_HIP_TARGET_SM PROPERTY HELPSTRING "List of HIP GPU architecture codes to compile for (without the sm_ prefix)")
    set_property(CACHE GMX_HIP_TARGET_SM PROPERTY TYPE STRING)
endif()

# default flags
list(APPEND _HIP_CLANG_FLAGS "-x hip" "-ffast-math" "-fhip-flush-denormals-to-zero")
# Workaround for clang>=9 (Bug 45533). No HIP file uses OpenMP.
list(APPEND _HIP_CLANG_FLAGS "-fno-openmp")
# HIP toolkit
list(APPEND _HIP_CLANG_FLAGS "--hip-path=${HIP_TOOLKIT_ROOT_DIR}")
# codegen flags
list(APPEND _HIP_CLANG_FLAGS "${_HIP_CLANG_GENCODE_FLAGS}")
foreach(_flag ${_HIP_CLANG_FLAGS})
    set(GMX_HIP_CLANG_FLAGS "${GMX_HIP_CLANG_FLAGS} ${_flag}")
endforeach()

if (HIP_USE_STATIC_HIP_RUNTIME)
    set(GMX_HIP_CLANG_LINK_LIBS "hiprt_static")
else()
    set(GMX_HIP_CLANG_LINK_LIBS "hiprt")
endif()
set(GMX_HIP_CLANG_LINK_LIBS "${GMX_HIP_CLANG_LINK_LIBS}" "dl" "rt")
if (HIP_64_BIT_DEVICE_CODE)
    set(GMX_HIP_CLANG_LINK_DIRS "${HIP_TOOLKIT_ROOT_DIR}/lib64")
else()
    set(GMX_HIP_CLANG_LINK_DIRS "${HIP_TOOLKIT_ROOT_DIR}/lib")
endif()

gmx_test_clang_hip_support()
