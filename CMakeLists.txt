cmake_minimum_required(VERSION 3.13)
project(pointcloud_upsample)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED common filters visualization)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pointcloud_upsample main.cpp)
target_link_libraries(pointcloud_upsample ${PCL_LIBRARIES})

add_executable(parallel parallel.cpp)

