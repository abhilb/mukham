#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "Detection.h"

namespace mukham {
enum class DetectionStatus {
    Success,
    PreProcessFailed,
    DetectionFailed,
    PostProcessFailed,
};

template <typename FaceDetectorModel>
class FaceDetector {
   public:
    DetectionStatus Detect(const cv::Mat &input,
                           std::vector<Landmarks> &landmarks,
                           std::vector<BoundingBox> &bounding_box) {
        return static_cast<FaceDetectorModel>(*this).detect(input, landmarks,
                                                            bounding_box);
    }
};
}  // namespace mukham
