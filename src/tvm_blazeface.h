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

struct BlazeFaceResult {};

class TVM_Blazeface final {
   public:
    explicit TVM_Blazeface(fs::path& model_path) {
        can_execute = true;
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
            } catch (...) {
                can_execute = false;
            }
        }
    }

    bool detect(const cv::Mat& input_image,
                tvm_blazeface::BlazeFaceResult& result);

   private:
    int input_width = 128;
    int input_height = 128;

    tr::Module gmod;
    tr::PackedFunc set_input;
    tr::PackedFunc get_output;
    tr::PackedFunc run;

    tr::NDArray input_tensor;
    tr::NDArray output_tensor_1;
    tr::NDArray output_tensor_2;

    bool can_execute;
};
}  // namespace tvm_blazeface
