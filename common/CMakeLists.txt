cmake_minimum_required(VERSION 3.16)
project(common CXX)

# GCC Standard
set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})

# OpenVINO
find_package(OpenVINO REQUIRED)
message("OpenVINO_VERSION: " ${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR})

# common
file(GLOB COMMON_SRC src/*)

add_library(${PROJECT_NAME} STATIC ${COMMON_SRC})

target_include_directories(
    ${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ./include
)

target_link_libraries(
    ${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
)
