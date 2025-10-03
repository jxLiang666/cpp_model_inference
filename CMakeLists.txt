CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT("cpp_model_inference" 
VERSION 0.0.2
LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)            # 使用 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 必须使用指定标准

add_subdirectory(${CMAKE_SOURCE_DIR}/deps/onnxruntime)
link_libraries(onnxruntime)

add_subdirectory(${CMAKE_SOURCE_DIR}/deps/opencv_4.12.0)
link_libraries(opencv)

find_package(TBB REQUIRED)
link_libraries(TBB::tbb)

add_subdirectory(${CMAKE_SOURCE_DIR}/src/nerual_net)
add_subdirectory(${CMAKE_SOURCE_DIR}/demo)
