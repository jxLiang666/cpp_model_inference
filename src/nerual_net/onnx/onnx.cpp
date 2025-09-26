#include <iostream>
#include <filesystem>
#include "onnx.h"
namespace fs = std::filesystem;
namespace nn {
Onnx::Onnx(const std::string &_model_path) : NerualNetwork() {
    model_path_ = _model_path;
    name_ = fs::path(_model_path).filename().string();

    init();
};
int Onnx::init() {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, name_.c_str());
    std::cout<<"model path: "<<model_path_<<std::endl;
    session_options_ = Ort::SessionOptions();
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);

    num_input_nodes_ = session_.GetInputCount();
    std::cout << "-- num_input_nodes: " << num_input_nodes_ << std::endl;
    input_node_names_.reserve(num_input_nodes_);
    input_types_.reserve(num_input_nodes_);
    input_node_dims_.reserve(num_input_nodes_);
    input_shape_.reserve(num_input_nodes_);
    for (size_t i = 0; i < num_input_nodes_; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_node_names_.push_back(name.get());
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_types_.push_back(tensor_info.GetElementType());
        input_node_dims_.push_back(tensor_info.GetShape());
        input_shape_.push_back(tensor_info.GetShape());
    }
    for (size_t i = 0; i < num_input_nodes_; ++i) {
        auto name = input_node_names_[i];
        std::cout << "Input name " << i << " : " << name << std::endl;
        auto type = input_types_[i];
        std::cout << "Input type " << i << " : " << type << std::endl;
        auto dim = input_node_dims_[i];
        std::cout << "Input dim " << i << " : ";
        for (size_t j = 0; j < dim.size(); ++j) {
            std::cout << dim[j] << " ";
        }
        std::cout << std::endl<<std::endl;
    }

    num_output_nodes_ = session_.GetOutputCount();
    std::cout << "-- num_output_nodes: " << num_output_nodes_ << std::endl;
    output_node_names_.reserve(num_output_nodes_);
    output_types_.reserve(num_output_nodes_);
    output_node_dims_.reserve(num_output_nodes_);
    output_shape_.reserve(num_output_nodes_);

    for (size_t i = 0; i < num_output_nodes_; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_node_names_.push_back(name.get());
        auto type_info = session_.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_types_.push_back(tensor_info.GetElementType());
        output_node_dims_.push_back(tensor_info.GetShape());
        output_shape_.push_back(tensor_info.GetShape());
    }
    for (size_t i = 0; i < num_output_nodes_; ++i) {
        auto name = output_node_names_[i];
        std::cout << "Output name " << i << " : " << name << std::endl;
        auto type = output_types_[i];
        std::cout << "Output type " << i << " : " << type << std::endl;
        auto dim = output_node_dims_[i];
        std::cout << "Output dim " << i << " : ";
        for (size_t j = 0; j < dim.size(); ++j) {
            std::cout << dim[j] << " ";
        }
        std::cout << std::endl<<std::endl;
    }

    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    return 0;
}

int Onnx::deinit() {
    return 0;
}
int Onnx::infer() {
    return 0;
}
int Onnx::preprocess() {
    return 0;
}
int Onnx::process() {
    return 0;
}
int Onnx::postprocess() {
    return 0;
}
Onnx::~Onnx() {
    deinit();
}
}  // namespace nn