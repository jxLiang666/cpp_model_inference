#pragma once
#include <string>
#include <vector>
#include "net_data.h"
#include "nn_config.h"
namespace nn {
class NerualNetworkBase {
public:
    NerualNetworkBase() = default;
    NerualNetworkBase(NerualNetworkBase &&) = default;
    NerualNetworkBase(const NerualNetworkBase &) = default;
    NerualNetworkBase &operator=(NerualNetworkBase &&) = default;
    NerualNetworkBase &operator=(const NerualNetworkBase &) = default;
    virtual ~NerualNetworkBase() = default;
    virtual int infer(_IN std::vector< std::vector< NetData > > &_input, _OUT std::vector< std::vector< NetData > > &_output) = 0;

protected:
    int         pipeline(_IN std::vector< std::vector< NetData > > &_input, _OUT std::vector< std::vector< NetData > > &_output);  ///< 封装整个前处理到后处理流程
    virtual int init() = 0;
    virtual int deinit() = 0;
    virtual int preprocess(_IN std::vector< std::vector< NetData > > &_input) = 0;
    virtual int process(_OUT std::vector< std::vector< NetData > > &_output) = 0;
    virtual int postprocess(_IN_OUT std::vector< std::vector< NetData > > &_output) = 0;

protected:
    std::string name_;        ///< 网络名称
    std::string model_path_;  ///< 模型路径

    std::vector< std::vector< int64_t > > input_shape_;           ///< 输入的每个维度
    std::vector< size_t >                 input_element_counts_;  ///< 输入总元素个数
    std::vector< size_t >                 input_element_size_;    ///< 输入元素所占內存(byte)
    std::vector< size_t >                 input_size_;            ///< 输入所占总内存(byte)

    std::vector< std::vector< int64_t > > output_shape_;           ///< 输出的每个维度
    std::vector< size_t >                 output_element_counts_;  ///< 输出总元素个数
    std::vector< size_t >                 output_element_size_;    ///< 输出元素所占內存(byte)
    std::vector< size_t >                 output_size_;            ///< 输出所占总内存(byte)
};
}  // namespace nn
