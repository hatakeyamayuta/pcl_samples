cmake_minimum_required(VERSION 3.5)
project(pcl_lshape)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(PCL 1.11.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcl_lshape src/main.cpp )
target_link_libraries(pcl_lshape ${PCL_LIBRARIES})
