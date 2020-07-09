cmake_minimum_required(VERSION 3.5)
project(cluster)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-o3")

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(clusters src/main.cpp )
target_link_libraries (clusters ${PCL_LIBRARIES})
