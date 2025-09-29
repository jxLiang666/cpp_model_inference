#include <execution>

#include "stdc.h"

namespace nn {
Stdc::Stdc(const std::string &_model_path) : Onnx(_model_path) {};
Stdc::~Stdc() {};
int Stdc::infer(std::vector< std::vector< NetData > > &_input, std::vector< std::vector< NetData > > &_output) {
    pipeline(_input, _output);
    // 从int64_t 转成 uint8_t
    using T = uint8_t;
    for (size_t i = 0; i < _output.size(); ++i) {
        for (size_t j = 0; j < _output[i].size(); ++j) {
            auto data = static_cast< int64_t * >(_output[i][j].getData());
            auto net_data = NetData(output_element_counts_[j] * sizeof(T));
            auto net_data_ptr = static_cast< T * >(net_data.getData());
            std::transform(std::execution::par, data, data + output_element_counts_[j], net_data_ptr, [](int64_t &v) -> T {
                return static_cast< T >(std::clamp(v, int64_t(0), int64_t(255)));
            });
            _output[i][j] = std::move(net_data);
        }
    }
    return 0;
}
int Stdc::postprocess(std::vector< std::vector< NetData > > &_output) {
    return 0;
};
}  // namespace nn
