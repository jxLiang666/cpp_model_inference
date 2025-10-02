#include <opencv2/opencv.hpp>

#include "stdc_data_adapter.h"
#include "net_data_op.h"
namespace nn {
std::vector< std::vector< NetData > > StdcDataAdapter::doCreateInput(std::any &&_input) {
    using TupleType = std::tuple< cv::Mat & >;
    auto &&args = std::any_cast< TupleType & >(_input);

    std::vector< std::vector< nn::NetData > > input;

    cv::Mat img = std::get< 0 >(args);
    cv::resize(img, img, prep_size_);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1 / 255.0);

    size_t channels = img.channels();
    size_t height = img.rows;
    size_t width = img.cols;
    size_t size = img.total() * img.elemSize();

    std::vector< nn::NetData > input_s;
    input_s.emplace_back(size);
    void *data = input_s[0].getData();
    std::memcpy(data, img.data, size);
    NetDataOp::hwc2chw< float >(input_s[0], channels, height, width);
    input.push_back(std::move(input_s));
    return input;
}
std::any StdcDataAdapter::doCreateOutput(std::vector< std::vector< NetData > > &_output) {
    std::vector< cv::Mat > masks(_output.size() * _output[0].size());
    for (int i = 0; i < _output.size(); ++i) {
        auto &&vec_net_data = _output[i];
        for (int j = 0; j < vec_net_data.size(); ++j) {
            cv::Mat mask(prep_size_.height, prep_size_.width, CV_8U);
            std::memcpy(mask.data, vec_net_data[j].getData(), vec_net_data[j].getSize());
            cv::resize(mask, mask, ori_size_, 0, 0, cv::INTER_NEAREST);
            masks[i * vec_net_data.size() + j] = mask;
        }
    }
    return masks;
}
}  // namespace nn