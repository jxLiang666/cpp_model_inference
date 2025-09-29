#pragma once
#include "onnx.h"
#include "net_data.h"

namespace nn {
class Stdc : public Onnx {
public:
    Stdc(const std::string &_model_path);
    Stdc(Stdc &&) = default;
    Stdc(const Stdc &) = default;
    Stdc &operator=(Stdc &&) = default;
    Stdc &operator=(const Stdc &) = default;
    ~Stdc() override;

    int infer(_IN std::vector< std::vector< NetData > > &_input, _OUT std::vector< std::vector< NetData > > &_output) override;

protected:
    int postprocess(_IN_OUT std::vector< std::vector< NetData > > &_output) override;
};
}  // namespace nn