#pragma once

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include <opencv2/core/types.hpp>
#include <tuple>
#include <vector>

using namespace std;
using namespace dlib;

template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<
    con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<
    con<1, 9, 9, 1, 1,
        rcon5<rcon5<
            rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

namespace dlib_facedetect {
class DlibFaceDetectDnn {
   public:
    DlibFaceDetectDnn();
    std::vector<cv::Rect2d> DetectFace(cv::Mat& image);

   private:
    net_type net;
};

class DlibFaceDetectHog {
   public:
    DlibFaceDetectHog();
    std::vector<cv::Rect2d> DetectFace(cv::Mat& image);

   private:
    dlib::frontal_face_detector _detector;
};
}  // namespace dlib_facedetect

