cmake_minimum_required(VERSION 3.5)
project(nonamls)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-o3")

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(nomals_viewer src/main.cpp )
target_link_libraries (nomals_viewer ${PCL_LIBRARIES})
