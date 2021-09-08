
#include <filesystem>
#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "tvm_blazeface.h"

namespace fs = std::filesystem;

int main() {
    auto model_path = fs::current_path() / "face_detection_short_range.so";
    tvm_blazeface::TVM_Blazeface model(model_path);
    cv::Mat image = cv::imread("face_detect.bmp");
    auto boxes = model.DetectFace(image);
    return 0;
}

