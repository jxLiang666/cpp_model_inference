#include <iostream>
#include <filesystem>

#include "onnx.h"

namespace nn {
Onnx::Onnx(const std::string &_model_path) : NerualNetwork() {
    namespace fs = std::filesystem;
    model_path_ = _model_path;
    name_ = fs::path(_model_path).filename().string();

    init();
};
int Onnx::init() {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, name_.c_str());
    std::cout << "model path: " << model_path_ << std::endl;
    session_options_ = Ort::SessionOptions();
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);

    num_input_nodes_ = session_.GetInputCount();
    std::cout << "-- num_input_nodes: " << num_input_nodes_ << std::endl;
    input_node_names_.reserve(num_input_nodes_);
    input_types_.reserve(num_input_nodes_);
    input_element_counts_.reserve(num_input_nodes_);
    input_shape_.reserve(num_input_nodes_);
    for (size_t i = 0; i < num_input_nodes_; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_node_names_.push_back(name.get());
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_types_.push_back(tensor_info.GetElementType());
        input_element_counts_.push_back(tensor_info.GetElementCount());
        input_shape_.push_back(tensor_info.GetShape());
    }
    for (size_t i = 0; i < num_input_nodes_; ++i) {
        auto name = input_node_names_[i];
        std::cout << "Input name " << i << " : " << name << std::endl;
        auto count = input_element_counts_[i];
        std::cout << "Input elem count " << i << " : " << count << std::endl;
        auto type = input_types_[i];
        std::cout << "Input type " << i << " : " << type << std::endl;
        auto dim = input_shape_[i];
        std::cout << "Input shape " << i << " : ";
        for (size_t j = 0; j < dim.size(); ++j) {
            std::cout << dim[j] << " ";
        }
        std::cout << '\n'
                  << std::endl;
    }

    num_output_nodes_ = session_.GetOutputCount();
    std::cout << "-- num_output_nodes: " << num_output_nodes_ << std::endl;
    output_node_names_.reserve(num_output_nodes_);
    output_types_.reserve(num_output_nodes_);
    output_element_counts_.reserve(num_output_nodes_);
    output_shape_.reserve(num_output_nodes_);

    for (size_t i = 0; i < num_output_nodes_; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_node_names_.push_back(name.get());
        auto type_info = session_.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_types_.push_back(tensor_info.GetElementType());
        output_element_counts_.push_back(tensor_info.GetElementCount());
        output_shape_.push_back(tensor_info.GetShape());
    }
    for (size_t i = 0; i < num_output_nodes_; ++i) {
        auto name = output_node_names_[i];
        std::cout << "Output name " << i << " : " << name << std::endl;
        auto count = output_element_counts_[i];
        std::cout << "Output elem count " << i << " : " << count << std::endl;
        auto type = output_types_[i];
        std::cout << "Output type " << i << " : " << type << std::endl;
        auto dim = output_shape_[i];
        std::cout << "Output shape " << i << " : ";
        for (size_t j = 0; j < dim.size(); ++j) {
            std::cout << dim[j] << " ";
        }
        std::cout << '\n'
                  << std::endl;
    }

    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    return 0;
}

int Onnx::deinit() {
    return 0;
}
// int Onnx::infer(std::vector< NetData > &_input, std::vector< NetData > &_output) {
//     return 0;
// }
int Onnx::preprocess(std::vector< NetData > &_input) {
    return 0;
}
int Onnx::process() {
    return 0;
}
// int Onnx::postprocess(std::vector< NetData > &_output) {
//     return 0;
// }
Onnx::~Onnx() {
    deinit();
}
}  // namespace nn