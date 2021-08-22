#include "opencv_face_detection.h"

#include <opencv2/imgproc.hpp>

namespace opencv_facedetect {
OpenCVFaceDetectLBP::OpenCVFaceDetectLBP() {
    _detector.load("lbpcascade_frontalface_improved.xml");
}

std::vector<cv::Rect2d> OpenCVFaceDetectLBP::DetectFace(cv::Mat& input_image) {
    std::vector<cv::Rect> faces;
    cv::Mat gray_image;
    cv::cvtColor(input_image, gray_image, cv::COLOR_RGB2GRAY);

    cv::equalizeHist(gray_image, gray_image);

    _detector.detectMultiScale(gray_image, faces, 1.1, 2,
                               cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    std::vector<cv::Rect2d> result;
    for (auto& face : faces)
        result.push_back(cv::Rect2d(face.x, face.y, face.width, face.height));

    return result;
}
}  // namespace opencv_facedetect
