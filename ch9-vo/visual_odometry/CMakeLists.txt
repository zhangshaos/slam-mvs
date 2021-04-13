cmake_minimum_required(VERSION 2.8)
project(vo)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_path(SOPHUS_INCLUDE_DIRS "sophus/common.hpp")

include_directories(${SOPHUS_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(test)