#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include "nn_config.h"
namespace nn {
namespace tool {
    class Box {
    public:
        explicit Box() : x1_(0), y1_(0), x2_(0), y2_(0), w_(0), h_(0), center_({0, 0}), score_(0) {};
        Box(Box &&) noexcept = default;
        Box(const Box &) = default;
        Box &operator=(Box &&) noexcept = default;
        Box &operator=(const Box &) = default;
        ~Box() = default;
        Box(float _x1, float _y1, float _x2, float _y2) : Box(_x1, _y1, _x2, _y2, 0) {}
        Box(float _x1, float _y1, float _x2, float _y2, float _score) : x1_(_x1), y1_(_y1), x2_(_x2), y2_(_y2), w_(_x2 - _x1), h_(_y2 - _y1), center_({_x1 / 2 + _x2 / 2, _y1 / 2 + _y2 / 2}), score_(_score) {};

        inline std::array< float, 4 > xyxy() const { return {x1_, y1_, x2_, y2_}; }
        inline std::array< float, 4 > xywh() const { return {x1_, y1_, w_, h_}; }
        inline float                  area() const { return w_ * h_; }
        inline std::array< float, 2 > getCenter() const { return center_; }
        inline float                  getScore() const { return score_; }
        float                         iou(_IN const Box &_box) const;
        static float                  iou(_IN const Box &_box1, _IN const Box &_box2);
        static std::vector< Box >     nms(_IN const std::vector< Box > &_boxes, _IN const float _iou_th, _OUT std::vector< size_t > &_idx);
        static std::vector< Box >     nms(_IN const std::vector< Box > &_boxes, _IN const float _iou_th);

    protected:
        float                  x1_;      ///< 左上横坐标
        float                  y1_;      ///< 左上纵坐标
        float                  x2_;      ///< 右下横坐标
        float                  y2_;      ///< 右下纵坐标
        float                  w_;       ///< 宽
        float                  h_;       ///< 高
        std::array< float, 2 > center_;  ///< 中心坐标
        float                  score_;   ///< 分数
    };
}  // namespace tool
}  // namespace nn