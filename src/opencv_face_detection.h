#pragma once

#include <vector>

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/objdetect.hpp"

namespace opencv_facedetect {
class OpenCVFaceDetectLBP {
   public:
    OpenCVFaceDetectLBP();
    std::vector<cv::Rect2d> DetectFace(cv::Mat& image);

   private:
    cv::CascadeClassifier _detector;
};

class OpenCVFaceDetectTF {
   public:
    OpenCVFaceDetectTF();
    std::vector<cv::Rect2d> DetectFace(cv::Mat& image);

   private:
    cv::dnn::Net _detector;
};

}  // namespace opencv_facedetect
