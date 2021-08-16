#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "FaceDetector.h"
#include "spdlog/spdlog"

namespace mukham {
class UltraFaceModel : public FaceDetector<UltraFaceModel> {
public:
  DetectionStatus detect(const cv::Mat &input,
                         std::vector<Landmarks> &landmarks,
                         std::vector<BoundingBox> &bounding_box);

private:
  void pre_process();
  void post_process();
};
} // namespace mukham
