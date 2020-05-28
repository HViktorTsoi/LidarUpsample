cmake_minimum_required(VERSION 3.13)
project(pointcloud_upsample)

set(CMAKE_CXX_STANDARD 14)
#
## python and pybind11
#set(PYTHON_INCLUDE_DIRS "/home/hviktortsoi/miniconda3/include/;/home/hviktortsoi/miniconda3/include/python3.7m")
#set(PYTHON_LIBARIES "/home/hviktortsoi/miniconda3/lib/")
#
#message("PYTHON INCLUDE DIRS: " ${PYTHON_INCLUDE_DIRS})
#message("PYTHON LIBRARY: " ${PYTHON_LIBARIES})
#
#set(CMAKE_PREFIX_PATH "/home/hviktortsoi/miniconda3/")

#add_subdirectory(/home/hviktortsoi/miniconda3/include/python3.7m/pybind11)

# pybind
set(pybind11_DIR "/home/hviktortsoi/miniconda3/share/cmake/pybind11")
#include_directories(${PYTHON_INCLUDE_DIRS})
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

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
add_executable(pointcloud_upsample main.cpp upsample.hpp)
target_link_libraries(pointcloud_upsample ${PCL_LIBRARIES})

## compile mls upsample library

add_library(upsample_ext SHARED upsample.cpp upsample.hpp)
set_target_properties(upsample_ext PROPERTIES PREFIX "")
target_link_libraries(upsample_ext pybind11::module ${PCL_LIBRARIES})
#pybind11_add_module(upsample_ext mls_upsample.cpp)
