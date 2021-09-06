#pragma once

#include <filesystem>

#include "dlpack/dlpack.h"
#include "opencv2/core.hpp"
#include "tvm/runtime/module.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/packed_func.h"

namespace tvm_blazeface {
namespace tr = tvm::runtime;
namespace fs = std::filesystem;

struct BlazeFaceResult {
    cv::Rect2i bounding_box;
    std::array<cv::Point2d, 5> key_points;
};

struct SSDOptions {
    int num_layers;
    double min_scale;
    double max_scale;
    int input_size_height;
    int input_size_width;
    double anchor_offset_x;
    double anchor_offset_y;
    std::array<int, 4> strides;
    double aspect_ratios;
    bool fixed_anchor_size;
};

struct TensorToBoxesOptions {
    int num_classes;
    int num_boxes;
    int num_coords;
    int box_coord_offset;
    int keypoint_coord_offset;
    int num_keypoints;
    int num_values_per_keypoint;
    bool sigmoid_score;
    double score_clipping_thresh;
    bool reverse_output_order;
    double x_scale;
    double y_scale;
    double h_scale;
    double w_scale;
    double min_score_thresh;
};

class TVM_Blazeface final {
   public:
    explicit TVM_Blazeface(fs::path& model_path, int batch_size = 1) {
        can_execute = true;
        anchor_options = {.num_layers = 4,
                          .min_scale = 0.1484375,
                          .max_scale = 0.75,
                          .input_size_height = 128,
                          .input_size_width = 128,
                          .anchor_offset_x = 0.5,
                          .anchor_offset_y = 0.5,
                          .strides = {8, 16, 16, 16},
                          .aspect_ratios = 1.0,
                          .fixed_anchor_size = true};

        box_options = {.num_classes = 1,
                       .num_boxes = 896,
                       .num_coords = 16,
                       .box_coord_offset = 0,
                       .keypoint_coord_offset = 4,
                       .num_keypoints = 6,
                       .num_values_per_keypoint = 2,
                       .sigmoid_score = true,
                       .score_clipping_thresh = 100.0,
                       .reverse_output_order = true,
                       .x_scale = 128.0,
                       .y_scale = 128.0,
                       .h_scale = 128.0,
                       .w_scale = 128.0,
                       .min_score_thresh = 0.5};

        auto file_exists = fs::exists(model_path);
        auto has_file_name = model_path.has_filename();
        auto has_extension = model_path.has_extension();
        if (file_exists && has_file_name && has_extension) {
            try {
                tr::Module mod_factory =
                    tr::Module::LoadFromFile(model_path.string());
                DLDevice dev{kDLCPU,
                             0};  //@todo: Add option to choose the device type
                gmod = mod_factory.GetFunction("default")(dev);
                set_input = gmod.GetFunction("set_input");
                get_output = gmod.GetFunction("get_output");
                run = gmod.GetFunction("run");

                input_tensor = tr::NDArray::Empty(
                    {batch_size, anchor_options.input_size_width,
                     anchor_options.input_size_height, 3},
                    DLDataType{kDLFloat, 32, 1}, dev);

                output_tensor_1 = tr::NDArray::Empty(
                    {batch_size, box_options.num_boxes, box_options.num_coords},
                    DLDataType{kDLFloat, 32, 1}, dev);

                output_tensor_2 =
                    tr::NDArray::Empty({batch_size, box_options.num_boxes, 1},
                                       DLDataType{kDLFloat, 32, 1}, dev);

            } catch (...) {
                can_execute = false;
            }
        }

        // Set the post processing options
        _get_anchor_boxes();
    }

    std::vector<cv::Rect2d> DetectFace(const cv::Mat& input_image);

    bool CanExecute() const { return can_execute; }

   private:
    void _get_anchor_boxes();
    void _decode_boxes(std::unique_ptr<float[]> raw_boxes,
                       std::unique_ptr<float[]> raw_scores,
                       std::vector<cv::Rect2d>& boxes);
    void _nms(const std::vector<cv::Rect2d>& boxes,
              const std::vector<float>& scores,
              std::vector<cv::Rect2d>& output);
    double _overlap_similarity(cv::Rect2d& box1, cv::Rect2d& box2);
    cv::Rect2d _get_intesection(const cv::Rect2d& box1, const cv::Rect2d& box2);

    int input_width = 128;
    int input_height = 128;
    int num_boxes = 896;
    int batch_size = 1;
    double min_supression_threshold = 0.3;

    tr::Module gmod;
    tr::PackedFunc set_input;
    tr::PackedFunc get_output;
    tr::PackedFunc run;

    tr::NDArray input_tensor;
    tr::NDArray output_tensor_1;
    tr::NDArray output_tensor_2;

    SSDOptions anchor_options;
    std::vector<std::pair<double, double>> anchors;

    TensorToBoxesOptions box_options;
    bool can_execute;
};
}  // namespace tvm_blazeface
