#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "nn_config.h"
namespace nn {
namespace tool {
    class Vis {
    public:
        Vis() = delete;
        ~Vis() = delete;
        Vis(const Vis &) = delete;
        Vis &operator=(const Vis &) = delete;
        Vis(Vis &&) = delete;
        Vis &operator=(Vis &&) = delete;

        static void setVisRoot(_IN const std::string &_root);                                                ///< 设置可视化工具保存目录，可以多次设置
        static void saveImg(_IN const cv::Mat &_img, _IN const std::string &_name, _IN bool _2bgr = false);  ///< 保存图片
        static void saveMask(_IN const cv::Mat &_mask, _IN const std::string &_name);                        ///< 保存掩码

    private:
        static void        makeDir(const std::string &_dir);
        static std::string getNow();

    private:
        static std::string              vis_root_;
        static std::vector< cv::Vec3b > colors_;
    };
}  // namespace tool
}  // namespace nn