#pragma once
#include <string>
#include <vector>
#include "net_data.h"
namespace nn {
class NerualNetwork {
public:
    NerualNetwork() = default;
    NerualNetwork(NerualNetwork &&) = default;
    NerualNetwork(const NerualNetwork &) = default;
    NerualNetwork &operator=(NerualNetwork &&) = default;
    NerualNetwork &operator=(const NerualNetwork &) = default;
    virtual ~NerualNetwork() = default;
    int run(std::vector< NetData > &_input, std::vector< NetData > &_output);

    virtual int init() = 0;
    virtual int deinit() = 0;
    virtual int infer(std::vector< NetData > &_input, std::vector< NetData > &_output) = 0;
    virtual int preprocess(std::vector< NetData > &_input) = 0;
    virtual int process() = 0;
    virtual int postprocess(std::vector< NetData > &_output) = 0;

protected:
    std::string                           name_;
    std::string                           model_path_;
    std::vector< std::vector< int64_t > > input_shape_;
    std::vector< std::vector< int64_t > > output_shape_;
};
}  // namespace nn
