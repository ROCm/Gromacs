#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2011,2012,2013,2014,2015 by the GROMACS development team.
# Copyright (c) 2016,2017,2018,2019,2020 by the GROMACS development team.
# Copyright (c) 2021, by the GROMACS development team, led by
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

if (NOT GMX_BUILD_UNITTESTS)
    gmx_add_missing_tests_notice("Unit tests have not been run. You need to set GMX_BUILD_UNITTESTS=ON if you want to build and run them.")
    return()
endif()

if (GMX_GPU_CUDA)
    # Work around FindCUDA that prevents using target_link_libraries()
    # with keywords otherwise...
    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
    if (NOT GMX_CLANG_CUDA)
        gmx_cuda_add_library(testutils)
    else()
        set_source_files_properties(test_device.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
        gmx_compile_cuda_file_with_clang(test_device.cpp)
        add_library(testutils STATIC ${TESTUTILS_SOURCES})
        target_link_libraries(testutils PRIVATE ${GMX_CUDA_CLANG_LINK_LIBS})
    endif()
    target_link_libraries(testutils PRIVATE ${CUDA_CUFFT_LIBRARIES})
elseif (GMX_GPU_HIP)
    if (NOT DEFINED ROCM_PATH)
        if (NOT DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "/opt/rocm")
        else()
            set(ROCM_PATH $ENV{ROCM_PATH})
        endif()
    endif()
    check_hip_path()
    get_property(HIP_ADD_LIBRARY_FOUND GLOBAL PROPERTY GMX_HIP_ADD_LIBRARY_FOUND)
    if(NOT HIP_ADD_LIBRARY_FOUND)
        list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake)
        # set(CMAKE_MODULE_PATH "${ROCM_PATH}/cmake" ${CMAKE_MODULE_PATH})
        find_package(HIP)
	    set_property(GLOBAL PROPERTY GMX_HIP_ADD_LIBRARY_FOUND true)
        if(HIP_FOUND)
            message(STATUS "Found HIP: ${HIP_VERSION}")
        else()
            message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
        endif()
    endif()

    string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)
    if("${build_type}" STREQUAL "release")
	    hip_add_library(testutils test_device.cpp HIPCC_OPTIONS "-fPIC -fno-gpu-rdc -std=c++17 -ggdb -ffast-math -DNDEBUG" NVCC_OPTIONS)
    else()
	    hip_add_library(testutils test_device.cpp HIPCC_OPTIONS "-fPIC -fno-gpu-rdc -std=c++17 -ggdb -ffast-math" NVCC_OPTIONS)
    endif()

    if (GMX_GPU_USE_VKFFT)
        target_link_libraries(testutils PRIVATE VkFFT)
    endif()    
    target_link_libraries(testutils PUBLIC roc::rocfft hip::hipfft hip::host)
else()
    add_library(testutils STATIC ${UNITTEST_TARGET_OPTIONS})
endif()

# Module interface / provided facilities
target_include_directories(testutils PUBLIC include)

# Executable targets for tests based on `testutils` acquire the source for
# their entry point from unittest_main.cpp when linking to the `testutils` target.
target_sources(testutils INTERFACE unittest_main.cpp)

target_sources(testutils PRIVATE
               cmdlinetest.cpp
               conftest.cpp
               filematchers.cpp
               interactivetest.cpp
               loggertest.cpp
               mpi_printer.cpp
               mpitest.cpp
               refdata.cpp
               refdata_xml.cpp
               simulationdatabase.cpp
               stdiohelper.cpp
               stringtest.cpp
               testasserts.cpp
               testfilemanager.cpp
               testfileredirector.cpp
               test_device.cpp
               test_hardware_environment.cpp
               testinit.cpp
               testmatchers.cpp
               testoptions.cpp
               textblockmatchers.cpp
               tprfilegenerator.cpp
               trajectoryreader.cpp
               xvgtest.cpp
               )

if(HAVE_TINYXML2)
    target_include_directories(testutils SYSTEM PRIVATE ${TinyXML2_INCLUDE_DIR})
    target_link_libraries(testutils PRIVATE ${TinyXML2_LIBRARIES})
else()
    target_include_directories(testutils SYSTEM BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/src/external/tinyxml2)
    target_sources(testutils PRIVATE ${CMAKE_SOURCE_DIR}/src/external/tinyxml2/tinyxml2.cpp)
endif()

target_include_directories(testutils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (GMX_GPU_SYCL)
    add_sycl_to_target(
        TARGET testutils
        SOURCES test_device.cpp
        )
endif()

gmx_target_compile_options(testutils)
target_compile_definitions(testutils PRIVATE HAVE_CONFIG_H)
target_include_directories(testutils SYSTEM BEFORE PRIVATE ${PROJECT_SOURCE_DIR}/src/external/thread_mpi/include)
target_link_libraries(testutils PRIVATE libgromacs ${GMX_COMMON_LIBRARIES} gmock)
target_link_libraries(testutils PUBLIC common)

# GROMACS module dependencies.
# Note that testutils conveys transitive dependencies on some modules.
# TODO: Explicitly link specific modules with minimal exposure.
target_link_libraries(testutils PUBLIC legacy_modules)

add_subdirectory(tests)
