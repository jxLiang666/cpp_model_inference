#pragma once
#include "nerual_network_base.h"
namespace nn {
class DebugNet : public NerualNetworkBase {
public:
    DebugNet(const std::string &_model_path) {}
    DebugNet(DebugNet &&) = default;
    DebugNet(const DebugNet &) = default;
    DebugNet &operator=(DebugNet &&) = default;
    DebugNet &operator=(const DebugNet &) = default;
    ~DebugNet() = default;

protected:
    int init() override;
    int deinit() override;
    int infer(_IN std::vector< std::vector< NetData > > &_input, _OUT std::vector< std::vector< NetData > > &_output) override;
    int preprocess(_IN std::vector< std::vector< NetData > > &_input) override;
    int process(_OUT std::vector< std::vector< NetData > > &_output) override;
    int postprocess(_IN_OUT std::vector< std::vector< NetData > > &_output) override;
};
}  // namespace nn