#pragma once

#include <opencv2/core.hpp>

template <typename T>
double get_iou(const cv::Rect_<T>& b1, const cv::Rect_<T>& b2) {
    double iou = 0;

    auto area1 = b1.area();
    auto area2 = b2.area();

    auto xx = (std::max)(b1.x, b2.x);
    auto yy = (std::max)(b1.y, b2.y);
    auto aa = (std::min)(b1.x + b1.width, b2.x + b2.width);
    auto bb = (std::min)(b1.y + b1.height, b2.y + b2.height);

    auto intersection_area =
        ((std::max<T>)(0, aa - xx)) * ((std::max<T>)(0, bb - yy));
    auto union_area = area1 + area2 - intersection_area;

    iou = intersection_area / union_area;

    return iou;
}

