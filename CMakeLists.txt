cmake_minimum_required(VERSION 3.16)
project(openvino_sample CXX)

# GCC Standard
set(CMAKE_CXX_STANDARD 17)

# common
add_subdirectory(common)

# human-pose-estimation-0007
add_subdirectory(sample/human-pose-estimation-0007)

# person-detection-0303
add_subdirectory(sample/person-detection-0303)

# vehicle-detection-0202
add_subdirectory(sample/vehicle-detection-0202)
