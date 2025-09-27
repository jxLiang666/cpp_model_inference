#pragma once
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include "nerual_network.h"
#include "box.h"

namespace nn {
class Onnx : public NerualNetwork {
public:
    Onnx(const std::string &_model_path);
    Onnx(Onnx &&) = default;
    Onnx(const Onnx &) = default;
    Onnx &operator=(Onnx &&) = default;
    Onnx &operator=(const Onnx &) = default;
    virtual ~Onnx() override;

    virtual int init() override;
    virtual int deinit() override;
    // virtual int infer(std::vector< NetData > &_input, std::vector< NetData > &_output) override;
    virtual int preprocess(std::vector< NetData > &_input) override;
    virtual int process() override;
    // virtual int postprocess(std::vector< NetData > &_output) override;

private:
    Ort::Env                         env_{nullptr};
    Ort::MemoryInfo                  memory_info_{nullptr};
    Ort::SessionOptions              session_options_{nullptr};
    Ort::Session                     session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    size_t                                   num_input_nodes_;
    std::vector< std::string >               input_node_names_;
    std::vector< ONNXTensorElementDataType > input_types_;
    std::vector< size_t >                    input_element_counts_;
    std::vector< Ort::Value >                input_tensors_;

    size_t                                   num_output_nodes_;
    std::vector< std::string >               output_node_names_;
    std::vector< ONNXTensorElementDataType > output_types_;
    std::vector< size_t >                    output_element_counts_;
    std::vector< Ort::Value >                output_tensors_;
};
}  // namespace nn