CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT("cpp_model_inference" 
VERSION 0.0.1
LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)            # 使用 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 必须使用指定标准

add_subdirectory(./deps/onnxruntime)
link_libraries(onnxruntime)

add_subdirectory(./deps/opencv_4.12.0)
link_libraries(opencv)

find_package(TBB REQUIRED)
link_libraries(TBB::tbb)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/box)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/nerual_network)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/nerual_network_base)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/onnx)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/net_data)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/visualizer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/stdc_net)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/config)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/net_data_op)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/net_debug_tool)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/data_adapter)

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/box Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/nerual_network Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/nerual_network_base Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/onnx Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/visualizer Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/stdc_net Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/net_data_op Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/net_debug_tool Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/data_adapter Allsrc)

add_executable(main main.cpp ${Allsrc})
