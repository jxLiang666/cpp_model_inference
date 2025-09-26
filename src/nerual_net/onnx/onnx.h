#pragma once
#include "nerual_network.h"
#include "box.h"
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
    virtual int postprocess();

private:
    Ort::Env                         env_{nullptr};
    Ort::MemoryInfo                  memory_info_{nullptr};
    Ort::SessionOptions              session_options_{nullptr};
    Ort::Session                     session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    size_t                                   num_input_nodes_;
    std::vector< std::string >               input_node_names_;
    std::vector< ONNXTensorElementDataType > input_types_;
    std::vector< std::vector< int64_t > >    input_node_dims_;
    std::vector< Ort::Value >                input_tensors_;

    size_t                                   num_output_nodes_;
    std::vector< std::string >               output_node_names_;
    std::vector< ONNXTensorElementDataType > output_types_;
    std::vector< std::vector< int64_t > >    output_node_dims_;
    std::vector< Ort::Value >                output_tensors_;
};
}  // namespace nn