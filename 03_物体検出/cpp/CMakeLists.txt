cmake_minimum_required (VERSION 2.8.12)
project(person_detection CXX)

set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -Wall")

include_directories(
    /opt/intel/openvino/inference_engine/include
    /opt/intel/openvino/inference_engine/external/tbb/include
    /opt/intel/openvino/deployment_tools/ngraph/include
    /opt/intel/openvino/opencv/include
)
link_directories(
    /opt/intel/openvino/inference_engine/lib/intel64
    /opt/intel/openvino/inference_engine/external/tbb/lib
    /opt/intel/openvino/deployment_tools/ngraph/lib
    /opt/intel/openvino/opencv/lib
)

add_executable(person_detection person_detection.cpp)
target_link_libraries(
    person_detection
    inference_engine inference_engine_legacy
    opencv_core opencv_imgcodecs opencv_imgproc
)
