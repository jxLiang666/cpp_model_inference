#pragma once
#include <opencv2/opencv.hpp>
#include "data_adapter.h"
namespace nn {
class ImgDataAdapter : public DataAdapterBase {
protected:
    explicit ImgDataAdapter(cv::Size &_ori_size, cv::Size &_prep_size) : ori_size_(_ori_size), prep_size_(_prep_size) {};
    cv::Size ori_size_;
    cv::Size prep_size_;
};
}  // namespace nn