#pragma once

#include <filesystem>

#include "opencv2/core.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/imgproc.hpp"
#include "tvm/runtime/module.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/packed_func.h"

namespace mukham {

namespace fs = std::filesystem;
namespace tr = tvm::runtime;

using Vec21f = cv::Vec<float, 21>;

class DeeplabSegmentationModel {
   public:
    DeeplabSegmentationModel();
    void Segment(const cv::Mat& input_image, cv::Mat& output_image);

   private:
    void _preprocess(const cv::Mat& input, cv::Mat& output);
    void _postprocess(const cv::Mat& input, cv::Mat& output,
                      const cv::Size& output_size);
    void _infer(const cv::Mat& input, cv::Mat& output);

    bool model_loaded;

    tr::Module gmod;
    tr::PackedFunc set_input;
    tr::PackedFunc get_output;
    tr::PackedFunc run;

    tr::NDArray input_tensor;
    tr::NDArray output_tensor;
};
}  // namespace mukham
