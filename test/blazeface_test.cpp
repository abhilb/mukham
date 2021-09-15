#include <gtest/gtest.h>

#include <filesystem>

#include "opencv2/imgcodecs.hpp"
#include "tvm_blazeface.h"

namespace fs = std::filesystem;

TEST(BlazeFaceTest, TestCanExecute) {
    auto model_path = fs::current_path() / "dummy.so";
    tvm_blazeface::TVM_Blazeface model(model_path);
    EXPECT_EQ(model.CanExecute(), false);
}

TEST(BlazeFaceTest, TestNormalize) {
    cv::Mat image = cv::imread("face_detect.bmp");
    cv::Mat out_image;
    tvm_blazeface::PreprocessImage(image, out_image, cv::Size(128, 128), -1.0,
                                   1.0);
    EXPECT_EQ(out_image.rows, 128);
    EXPECT_EQ(out_image.cols, 128);

    double minval, maxval;
    auto vec = out_image.reshape(1, 1);
    cv::minMaxLoc(vec, &minval, &maxval);

    EXPECT_FLOAT_EQ(maxval, 0.77254903);
    EXPECT_FLOAT_EQ(minval, -0.7882353);
}
