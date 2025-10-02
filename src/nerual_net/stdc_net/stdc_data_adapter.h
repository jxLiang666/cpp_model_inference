#pragma once
#include "img_data_adapter.h"
namespace nn {
class StdcDataAdapter : public ImgDataAdapter {
public:
    explicit StdcDataAdapter(cv::Size _ori_size, cv::Size _prep_size) : ImgDataAdapter(_ori_size, _prep_size) {}

protected:
    std::vector< std::vector< NetData > > doCreateInput(std::any &&_input) override;
    std::any                              doCreateOutput(std::vector< std::vector< NetData > > &_output) override;
};
}  // namespace nn