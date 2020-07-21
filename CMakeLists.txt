cmake_minimum_required(VERSION 3.5)
project(pointcloud_upsample)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)
## python and pybind11
#set(PYTHON_INCLUDE_DIRS "${PYTHON_ROOT}/include/;${PYTHON_ROOT}/include/python${PYTHON_VERSION}m")
#set(PYTHON_LIBARIES "${PYTHON_ROOT}/lib/")
#
#message("PYTHON INCLUDE DIRS: " ${PYTHON_INCLUDE_DIRS})
#message("PYTHON LIBRARY: " ${PYTHON_LIBARIES})
#include_directories(${PYTHON_INCLUDE_DIRS})

## pybind
#set(pybind11_DIR "/home/hviktortsoi/miniconda3/share/cmake/pybind11")
##set(pybind11_DIR "${PYTHON_ROOT}/share/cmake/pybind11")
##message("Pybind11 find path:" ${pybind11_DIR})
#find_package(pybind11 REQUIRED)
#include_directories(${pybind11_INCLUDE_DIRS})

# pcl
#set(PCL_DIR "/media/hviktortsoi/302F52149E388942/BaiduNetdiskDownload/pcl-1.8.1/share/pcl-1.8")
find_package(PCL REQUIRED common filters visualization)
include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# open mp
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

# main
add_executable(pointcloud_upsample src/main.cpp src/upsample.hpp src/mls_lidar.hpp)
target_link_libraries(pointcloud_upsample ${PCL_LIBRARIES})

## compile mls upsample library
pybind11_add_module(upsample_ext src/upsample.cpp src/upsample.hpp)
set_target_properties(upsample_ext PROPERTIES PREFIX "")
set_target_properties(upsample_ext PROPERTIES SUFFIX ".so")
target_link_libraries(upsample_ext PRIVATE ${PYTHON_LIBARIES} ${PCL_LIBRARIES})
#
#add_library(upsample_ext SHARED upsample.cpp upsample.hpp)
#set_target_properties(upsample_ext PROPERTIES PREFIX "")
##target_link_libraries(upsample_ext pybind11::module ${PCL_LIBRARIES})
#target_link_libraries(upsample_ext ${PYTHON_LIBARIES} ${PCL_LIBRARIES})
