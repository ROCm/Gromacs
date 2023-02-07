find_path(roctracer_roctracer_INCLUDE_DIRS "roctracer/roctracer.h"
          HINTS ${roctracer_DIR}
          PATHS "$ENV{ROCM_ROOT}" "/opt/rocm"
          PATH_SUFFIXES "include")

find_library(roctracer_roctracer_LIBRARIES roctracer64 
             HINTS ${roctracer_DIR}
             PATHS "$ENV{ROCM_ROOT}" "/opt/rocm"
             PATH_SUFFIXES "lib" "lib64")

if(roctracer_roctracer_INCLUDE_DIRS
   AND EXISTS "${roctracer_roctracer_INCLUDE_DIRS}/roctracer/roctracer.h"
   AND roctracer_roctracer_LIBRARIES)
    file(STRINGS "${roctracer_roctracer_INCLUDE_DIRS}/roctracer/roctracer.h" roctracer_roctracer_H_MAJOR_VERSION REGEX "#define ROCTRACER_VERSION_MAJOR [0-9]+")
    file(STRINGS "${roctracer_roctracer_INCLUDE_DIRS}/roctracer/roctracer.h" roctracer_roctracer_H_MINOR_VERSION REGEX "#define ROCTRACER_VERSION_MINOR [0-9]+")
    string(REGEX REPLACE "#define ROCTRACER_VERSION_MAJOR ([0-9]+)" "\\1" roctracer_roctracer_MAJOR_VERSION "${roctracer_roctracer_H_MAJOR_VERSION}")
    string(REGEX REPLACE "#define ROCTRACER_VERSION_MINOR ([0-9]+)" "\\1" roctracer_roctracer_MINOR_VERSION "${roctracer_roctracer_H_MINOR_VERSION}")
    set(roctracer_roctracer_VERSION "${roctracer_roctracer_MAJOR_VERSION}.${roctracer_roctracer_H_MINOR_VERSION}")
    set(roctracer_roctracer_FOUND TRUE)
    add_library(roc::roctracer64 UNKNOWN IMPORTED)
    set_target_properties(roc::roctracer64 PROPERTIES IMPORTED_LOCATION "${roctracer_roctracer_LIBRARIES}")
    target_include_directories(roc::roctracer64 INTERFACE "${roctracer_roctracer_INCLUDE_DIRS}")
endif()

find_path(roctracer_roctx_INCLUDE_DIRS "roctracer/roctx.h"
          HINTS ${roctracer_DIR}
          PATHS "$ENV{ROCM_ROOT}" "/opt/rocm"
          PATH_SUFFIXES "include")

find_library(roctracer_roctx_LIBRARIES roctx64 
             HINTS ${roctracer_DIR}
             PATHS "$ENV{ROCM_ROOT}" "/opt/rocm"
             PATH_SUFFIXES "lib" "lib64")

if(EXISTS "${roctracer_roctx_INCLUDE_DIRS}/roctracer/roctx.h"
   AND roctracer_roctx_LIBRARIES)
    file(STRINGS "${roctracer_roctx_INCLUDE_DIRS}/roctracer/roctracer.h" roctracer_roctx_H_MAJOR_VERSION REGEX "#define ROCTX_VERSION_MAJOR [0-9]+")
    file(STRINGS "${roctracer_roctx_INCLUDE_DIRS}/roctracer/roctracer.h" roctracer_roctx_H_MINOR_VERSION REGEX "#define ROCTX_VERSION_MINOR [0-9]+")
    string(REGEX REPLACE "#define ROCTX_VERSION_MAJOR ([0-9]+)" "\\1" roctracer_roctx_MAJOR_VERSION "${roctracer_roctx_H_MAJOR_VERSION}")
    string(REGEX REPLACE "#define ROCTX_VERSION_MINOR ([0-9]+)" "\\1" roctracer_roctx_MINOR_VERSION "${roctracer_roctx_H_MINOR_VERSION}")
    set(roctracer_roctx_VERSION "${roctracer_roctx_MAJOR_VERSION}.${roctracer_roctx_H_MINOR_VERSION}")
    set(roctracer_roctx_FOUND TRUE)
    add_library(roc::roctx64 UNKNOWN IMPORTED)
    set_target_properties(roc::roctx64 PROPERTIES IMPORTED_LOCATION "${roctracer_roctx_LIBRARIES}")
    target_include_directories(roc::roctx64 INTERFACE "${roctracer_roctx_INCLUDE_DIRS}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(roctracer FOUND_VAR roctracer_FOUND
                                  REQUIRED_VARS
                                    roctracer_roctracer_INCLUDE_DIRS
                                    roctracer_roctracer_LIBRARIES
                                    roctracer_roctracer_FOUND
                                    roctracer_roctx_INCLUDE_DIRS
                                    roctracer_roctx_LIBRARIES
                                    roctracer_roctx_FOUND)

mark_as_advanced(roctracer_roctracer_INCLUDE_DIRS
                 roctracer_roctracer_LIBRARIES
                 roctracer_roctx_INCLUDE_DIRS
                 roctracer_roctx_LIBRARIES)
