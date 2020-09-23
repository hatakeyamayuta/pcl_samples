cmake_minimum_required(VERSION 3.5)
project(occlusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-o3")

find_package(PCL 1.11 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(oocclusion src/main.cpp )
target_link_libraries (occlusion ${PCL_LIBRARIES})
