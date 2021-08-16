
#include "dlib_face_detection.h"

#include <tuple>

#include "dlib/image_processing/frontal_face_detector.h"

namespace dlib_facedetect {

DlibFaceDetectDnn::DlibFaceDetectDnn() {
    deserialize("mmod_human_face_detector.dat") >> net;
}

std::vector<cv::Rect2d> DlibFaceDetectDnn::DetectFace(cv::Mat& image) {
    std::vector<cv::Rect2d> result;

    dlib::cv_image<dlib::rgb_pixel> dlib_image(image);
    dlib::matrix<rgb_pixel> dlib_matrix;
    dlib::assign_image(dlib_matrix, dlib_image);

    std::vector<dlib::mmod_rect> dets = net(dlib_matrix);
    for (auto&& d : dets) {
        result.push_back(cv::Rect2d(d.rect.left(), d.rect.top(), d.rect.width(),
                                    d.rect.height()));
    }

    return result;
}

DlibFaceDetectHog::DlibFaceDetectHog() {
    _detector = dlib::get_frontal_face_detector();
}

std::vector<cv::Rect2d> DlibFaceDetectHog::DetectFace(cv::Mat& image) {
    std::vector<cv::Rect2d> result;

    dlib::cv_image<dlib::rgb_pixel> dlib_image(image);

    std::vector<dlib::rectangle> dets = _detector(dlib_image);
    for (auto&& d : dets) {
        result.push_back(cv::Rect2d(d.left(), d.top(), d.width(), d.height()));
    }

    return result;
}
}  // namespace dlib_facedetect
