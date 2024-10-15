cmake_minimum_required(VERSION 3.16)
project(openvino_sample CXX)

# GCC Standard
set(CMAKE_CXX_STANDARD 17)

# common
add_subdirectory(common)

# person-detection-0303
add_subdirectory(sample/person-detection-0303)
