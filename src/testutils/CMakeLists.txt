#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2011,2012,2013,2014,2015 by the GROMACS development team.
# Copyright (c) 2016,2017,2018,2019,2020, by the GROMACS development team, led by
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

set(TESTUTILS_SOURCES
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
    xvgtest.cpp
    )

if(NOT HAVE_TINYXML2)
    list(APPEND TESTUTILS_SOURCES ../external/tinyxml2/tinyxml2.cpp)
endif()

if (GMX_GPU_CUDA)
    # Work around FindCUDA that prevents using target_link_libraries()
    # with keywords otherwise...
    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
    if (NOT GMX_CLANG_CUDA)
        gmx_cuda_add_library(testutils ${TESTUTILS_SOURCES})
    else()
        set_source_files_properties(test_device.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
        gmx_compile_cuda_file_with_clang(test_device.cpp)
        add_library(testutils STATIC ${TESTUTILS_SOURCES})
        target_link_libraries(testutils PRIVATE ${GMX_CUDA_CLANG_LINK_LIBS})
    endif()
    target_link_libraries(testutils PRIVATE ${CUDA_CUFFT_LIBRARIES})
elseif(GMX_GPU_HIP)
    if (NOT DEFINED ROCM_PATH)
        if (NOT DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "/opt/rocm")
        else()
            set(ROCM_PATH $ENV{ROCM_PATH})
        endif()
    endif()
    set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HIP_CLANG_PATH} ${HIP_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

    if(NOT DEFINED HIP_PATH)
	if(NOT DEFINED ENV{HIP_PATH})
		set(HIP_PATH ${ROCM_PATH}"/hip" CACHE PATH "Path to which HIP has been installed")
		set(HIP_CLANG_PATH ${ROCM_PATH}"/llvm/bin" CACHE PATH "Path to which HIP  clang has been installed")
	else()
	    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
	    set(HIP_CLANG_PATH ${ROCM_PATH}"/llvm/bin" CACHE PATH "Path to which HIP  clang has been installed")
	endif()
    endif()
    set_source_files_properties(test_device.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    get_property(HIP_ADD_LIBRARY_FOUND GLOBAL PROPERTY GMX_HIP_ADD_LIBRARY_FOUND)

    if (NOT HIP_ADD_LIBRARY_FOUND )
	    list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake)
        find_package(HIP QUIET)
    	if(HIP_FOUND)
	    message(STATUS "Found HIP: " ${HIP_VERSION})
	else()
	    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
	endif()
	set_property(GLOBAL PROPERTY GMX_HIP_ADD_LIBRARY_FOUND true)
    endif()

    if(NOT GMX_THREAD_MPI)
        find_package(MPI)
        if(MPI_C_FOUND)
            execute_process(
                COMMAND ${MPI_C_COMPILER} -showme:incdirs
                OUTPUT_VARIABLE  MPI_INCDIRS OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_VARIABLE   MPI_INCDIRS ERROR_STRIP_TRAILING_WHITESPACE)
            separate_arguments(MPI_INCDIRS)
            message(STATUS "MPI_INCDIRS: ${MPI_INCDIRS}")
            include_directories(SYSTEM ${MPI_INCDIRS})
        endif()
    endif()

    hip_add_library(testutils ${TESTUTILS_SOURCES} HIPCC_OPTIONS "-fPIC -fno-gpu-rdc -std=c++17 -ffast-math -DNDEBUG" NVCC_OPTIONS)
    target_link_libraries(testutils PRIVATE roc::rocfft)

else()
    add_library(testutils STATIC ${UNITTEST_TARGET_OPTIONS} ${TESTUTILS_SOURCES})
endif()

if (GMX_GPU_SYCL)
    set_source_files_properties(test_device.cpp
      PROPERTIES COMPILE_FLAGS "${SYCL_CXX_FLAGS}")
endif()

gmx_target_compile_options(testutils)
target_compile_definitions(testutils PRIVATE HAVE_CONFIG_H)
target_include_directories(testutils SYSTEM BEFORE PRIVATE ${PROJECT_SOURCE_DIR}/src/external/thread_mpi/include)
target_link_libraries(testutils PRIVATE libgromacs ${GMX_COMMON_LIBRARIES} gmock)

if(HAVE_TINYXML2)
    include_directories(SYSTEM ${TinyXML2_INCLUDE_DIR})
    target_link_libraries(testutils PRIVATE ${TinyXML2_LIBRARIES})
else()
    include_directories(BEFORE SYSTEM "../external/tinyxml2")
endif()

# TODO Use gmx_add_missing_tests_notice() instead of the messages below.
set(GMX_CAN_RUN_MPI_TESTS 1)
if (GMX_MPI)
    set(_an_mpi_variable_had_content 0)
    foreach(VARNAME MPIEXEC MPIEXEC_NUMPROC_FLAG MPIEXEC_PREFLAGS MPIEXEC_POSTFLAGS)
        # These variables need a valid value for the test to run
        # and pass, but conceivably any of them might be valid
        # with arbitrary (including empty) content. They can't be
        # valid if they've been populated with the CMake
        # find_package magic suffix/value "NOTFOUND", though.
        if (${VARNAME} MATCHES ".*NOTFOUND")
            message(STATUS "CMake variable ${VARNAME} was not detected to be a valid value. To test GROMACS correctly, check the advice in the install guide.")
            set(GMX_CAN_RUN_MPI_TESTS 0)
        endif()
        if (NOT VARNAME STREQUAL MPIEXEC AND ${VARNAME})
            set(_an_mpi_variable_had_content 1)
        endif()
    endforeach()
    if(_an_mpi_variable_had_content AND NOT MPIEXEC)
        message(STATUS "CMake variable MPIEXEC must have a valid value if one of the other related MPIEXEC variables does. To test GROMACS correctly, check the advice in the install guide.")
        set(GMX_CAN_RUN_MPI_TESTS 0)
    endif()
elseif (NOT GMX_THREAD_MPI)
    set(GMX_CAN_RUN_MPI_TESTS 0)
endif()

set(TESTUTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TESTUTILS_DIR ${TESTUTILS_DIR} PARENT_SCOPE)
set(GMX_CAN_RUN_MPI_TESTS ${GMX_CAN_RUN_MPI_TESTS} PARENT_SCOPE)

add_subdirectory(tests)
