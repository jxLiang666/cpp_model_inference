#pragma once
#include "nerual_network.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
namespace nn {
class Onnx : public NerualNetwork {
public:
    Onnx(const std::string &_model_path);
    Onnx(Onnx &&) = default;
    Onnx(const Onnx &) = default;
    Onnx &operator=(Onnx &&) = default;
    Onnx &operator=(const Onnx &) = default;
    virtual ~Onnx();

    virtual int init();
    virtual int deinit();
    virtual int infer();
    virtual int preprocess();
    virtual int process();

private:
    Ort::Env                         env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::SessionOptions              session_options_;
    Ort::Session                     session_;
    Ort::Value                       input_tensor_;
    Ort::Value                       output_tensor_;

    size_t                                   num_input_nodes_;
    std::vector< const char * >              input_node_names_;
    std::vector< std::vector< int64_t > >    input_node_dims_;
    std::vector< ONNXTensorElementDataType > input_types_;
    std::vector< OrtValue * >                input_tensors_;

    size_t                                   num_output_nodes_;
    std::vector< const char * >              output_node_names_;
    std::vector< std::vector< int64_t > >    output_node_dims_;
    std::vector< ONNXTensorElementDataType > output_types_;
    std::vector< OrtValue * >                output_tensors_;
};
}  // namespace nn