#include "opencv_face_detection.h"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

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

OpenCVFaceDetectTF::OpenCVFaceDetectTF() {
    const std::string config_file{"./opencv_face_detector.pbtxt"};
    const std::string weight_file{"./opencv_face_detector_uint8.pb"};
    _detector = cv::dnn::readNetFromTensorflow(weight_file, config_file);
}
std::vector<cv::Rect2d> OpenCVFaceDetectTF::DetectFace(cv::Mat& input_image) {
    std::vector<cv::Rect2d> faces;

    auto input_tensor = cv::dnn::blobFromImage(
        input_image, 255.0, cv::Size(300, 300), cv::Scalar(104, 117, 123));
    _detector.setInput(input_tensor, "data");
    auto detections = _detector.forward("detection_out");

    cv::Mat detection_mat(detections.size[2], detections.size[3], CV_32F,
                          detections.ptr<float>());
    for (int i = 0; i < detection_mat.rows; ++i) {
        auto score = detection_mat.at<float>(i, 2);

        if (score > 0.5) {
            cv::Rect2d face(cv::Point2d(detection_mat.at<float>(i, 3),
                                        detection_mat.at<float>(i, 4)),
                            cv::Point2d(detection_mat.at<float>(i, 5),
                                        detection_mat.at<float>(i, 6)));
            faces.push_back(face);
        }
    }
    return faces;
}
}  // namespace opencv_facedetect
