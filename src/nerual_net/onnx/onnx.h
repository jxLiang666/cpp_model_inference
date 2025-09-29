#pragma once
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include "nerual_network_base.h"
#include "box.h"

namespace nn {
class Onnx : public NerualNetworkBase {
public:
    Onnx(const std::string &_model_path);
    Onnx(Onnx &&) = default;
    Onnx(const Onnx &) = default;
    Onnx &operator=(Onnx &&) = default;
    Onnx &operator=(const Onnx &) = default;
    virtual ~Onnx() override;

protected:
    virtual int init() override;
    virtual int deinit() override;
    // virtual int infer(std::vector< NetData > &_input, std::vector< NetData > &_output) override;
    virtual int preprocess(_IN std::vector< std::vector< NetData > > &_input) override;
    virtual int process(_OUT std::vector< std::vector< NetData > > &_output) override;
    // virtual int postprocess(std::vector< NetData > &_output) override;

private:
    static int getONNXTensorElementDataTypeSize(_IN const ONNXTensorElementDataType &_type);

private:
    Ort::Env                         env_{nullptr};
    Ort::MemoryInfo                  memory_info_{nullptr};
    Ort::SessionOptions              session_options_{nullptr};
    Ort::Session                     session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    size_t                                   num_input_nodes_;     ///< 输入节点数量
    std::vector< std::string >               input_node_names_;    ///< 输入节点名字
    std::vector< const char * >              input_node_names_c_;  ///< 输入节点名字,c类型
    std::vector< ONNXTensorElementDataType > input_types_;         ///< 输入变量类型
    std::vector< std::vector< Ort::Value > > input_tensors_;       ///< 输入张量

    size_t                                   num_output_nodes_;     ///< 输出节点数量
    std::vector< std::string >               output_node_names_;    ///< 输出节点名字
    std::vector< const char * >              output_node_names_c_;  ///< 输出节点名字,c类型
    std::vector< ONNXTensorElementDataType > output_types_;         ///< 输出变量类型
    std::vector< std::vector< Ort::Value > > output_tensors_;       ///< 输出张量
};
}  // namespace nn