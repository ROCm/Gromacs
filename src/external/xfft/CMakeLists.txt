Project(XFFT)

file(GLOB XFFT_SOURCES src/*.cpp)
find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

set_source_files_properties(${XFFT_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)
if("${build_type}" STREQUAL "release")
	hip_add_library(xfft STATIC ${XFFT_SOURCES} HIPCC_OPTIONS "-fPIC -fno-gpu-rdc -std=c++14 -DNDEBUG" HCC_OPTIONS "" NVCC_OPTIONS)
else()
	hip_add_library(xfft STATIC ${XFFT_SOURCES} HIPCC_OPTIONS "-fPIC -fno-gpu-rdc -std=c++14" HCC_OPTIONS "" NVCC_OPTIONS)
endif()

target_include_directories(xfft 
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include/xfft>
	)

set(xfft_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
set(xfft_LIBRARIES ${CMAKE_BINARY_DIR}/lib/libxfft.a PARENT_SCOPE)

install(TARGETS xfft 
	LIBRARY DESTINATION lib/xfft
	)
