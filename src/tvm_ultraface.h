#pragma once

#include <filesystem>

#include "dlpack/dlpack.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "spdlog/spdlog.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/packed_func.h"

namespace fs = std::filesystem;
namespace tr = tvm::runtime;

namespace mukham
{
    struct PostProcessParams
    {
        int input_width;
        int input_height;
        double score_threshold;
        double iou_threshold;
    };

    class UltraFaceModel
    {
    public:
        UltraFaceModel(const fs::path &model_path);

    private:
        void _preprocess(const cv::Mat &input_image, cv::Mat &output_image);
        void _postprocess(const std::vector<double> &scores, const std::vector<cv::Rect2d> &boxes, const PostProcessParams &params)
            const int input_width = 320;
        const int input_height = 240;
        const int channels = 3;

        tr::Module gmod;
        tr::PackedFunc set_input;
        tr::PackedFunc get_output;
        tr::PackedFunc run;

        tr::NDArray input_tensor;
        tr::NDArray scores_tensor;
        tr::NDArray boxes_tensor;
    }
}