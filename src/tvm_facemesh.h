#pragma once

#include <chrono>
#include <filesystem>

#include "dlpack/dlpack.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "spdlog/spdlog.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/packed_func.h"

namespace tvm_facemesh {
namespace fs = std::filesystem;
namespace tr = tvm::runtime;

struct TVM_FacemeshResult {
    bool has_face;
    double face_score;
    std::vector<cv::Point2f> mesh;
};

class TVM_Facemesh {
   public:
    TVM_Facemesh(const fs::path& model_path, int batch_size = 1) {
        can_execute = true;
        _batch_size = batch_size;
        nr_positions = _batch_size * nr_landmarks;
        try {
            spdlog::info("Model: {}", model_path.string());
            if (!fs::exists(model_path)) {
                spdlog::info("Model exists: {}", fs::exists(model_path));
                can_execute = false;
            } else {
                tr::Module mod_factory =
                    tr::Module::LoadFromFile(model_path.string());
                //@todo: Add option to choose the device type
                DLDevice dev{kDLCPU, 0};
                gmod = mod_factory.GetFunction("default")(dev);
                set_input = gmod.GetFunction("set_input");
                get_output = gmod.GetFunction("get_output");
                run = gmod.GetFunction("run");

                input_tensor = tr::NDArray::Empty(
                    {batch_size, input_width, input_height, 3},
                    DLDataType{kDLFloat, 32, 1}, dev);
                output_tensor_1 = tr::NDArray::Empty(
                    {batch_size, 1, 1, 1404}, DLDataType{kDLFloat, 32, 1}, dev);
                output_tensor_2 = tr::NDArray::Empty(
                    {batch_size, 1, 1, 1}, DLDataType{kDLFloat, 32, 1}, dev);
            }
        } catch (...) {
            spdlog::error("Failed to create FaceMesh model object");
            can_execute = false;
        }
    }

    bool Detect(const std::vector<cv::Mat>& frame,
                std::vector<TVM_FacemeshResult>& result);

    bool Detect(const cv::Mat& frame, TVM_FacemeshResult& result);

   private:
    bool can_execute;
    int _batch_size = 1;
    const int input_width = 192;
    const int input_height = 192;
    const int channels = 3;
    const int nr_landmarks = 1404;
    int nr_positions = _batch_size * nr_landmarks;

    tr::Module gmod;
    tr::PackedFunc set_input;
    tr::PackedFunc get_output;
    tr::PackedFunc run;

    tr::NDArray input_tensor;
    tr::NDArray output_tensor_1;
    tr::NDArray output_tensor_2;
};
}  // namespace tvm_facemesh
