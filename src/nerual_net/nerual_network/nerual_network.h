#pragma once
#include <string>
#include <vector>
namespace nn {
class NerualNetwork {
public:
    NerualNetwork() = default;
    NerualNetwork(NerualNetwork &&) = default;
    NerualNetwork(const NerualNetwork &) = default;
    NerualNetwork &operator=(NerualNetwork &&) = default;
    NerualNetwork &operator=(const NerualNetwork &) = default;
    virtual ~NerualNetwork() = default;
    int run();

    virtual int init() = 0;
    virtual int deinit() = 0;
    virtual int infer() = 0;
    virtual int preprocess() = 0;
    virtual int process() = 0;
    virtual int postprocess() = 0;

protected:
    std::string                           name_;
    std::string                           model_path_;
    std::vector< std::vector< float > >   input_;
    std::vector< std::vector< float > >   output_;
    std::vector< std::vector< int64_t > > input_shape_;
    std::vector< std::vector< int64_t > > output_shape_;
};
}  // namespace nn
