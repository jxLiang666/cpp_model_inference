CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT("cpp_model_inference" 
VERSION 0.0.1
LANGUAGES C CXX)

add_subdirectory(./deps/onnxruntime)
link_libraries(onnxruntime)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/box)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/nerual_network)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/onnx)

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/box Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/nerual_network Allsrc)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/nerual_net/onnx Allsrc)

add_executable(main main.cpp onnxruntime ${Allsrc})
