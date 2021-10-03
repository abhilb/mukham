
#include "deeplab_segmentation.h"

#include "spdlog/spdlog.h"

namespace mukham {
DeeplabSegmentationModel::DeeplabSegmentationModel() {
    const fs::path model_path{"./lite-model_deeplabv3_1_metadata_2.so"};
    model_loaded = fs::exists(model_path);
    if (model_loaded) {
        try {
            tr::Module mod_factory =
                tr::Module::LoadFromFile(model_path.string());
            DLDevice dev{kDLCPU,
                         0};  //@todo: Add option to choose the device type
            gmod = mod_factory.GetFunction("default")(dev);
            set_input = gmod.GetFunction("set_input");
            get_output = gmod.GetFunction("get_output");
            run = gmod.GetFunction("run");

            input_tensor = tr::NDArray::Empty({1, 257, 257, 3},
                                              DLDataType{kDLFloat, 32, 1}, dev);
            output_tensor = tr::NDArray::Empty(
                {1, 257, 257, 21}, DLDataType{kDLFloat, 32, 1}, dev);
        } catch (...) {
            spdlog::error("Failed to load the deeplab v3 model");
            model_loaded = false;
        }
    }
}

void DeeplabSegmentationModel::Segment(const cv::Mat& input_image,
                                       cv::Mat& output_image) {
    cv::Mat preprocessed_image;
    cv::Mat model_output;

    _preprocess(input_image, preprocessed_image);
    _infer(preprocessed_image, model_output);
    _postprocess(model_output, output_image);
}

void DeeplabSegmentationModel::_preprocess(const cv::Mat& input,
                                           cv::Mat& output) {
    cv::Mat scaled_image;
    auto output_size = cv::Size(257, 257);
    cv::resize(input, scaled_image, output_size, cv::INTER_AREA);
    const double max_val = 1.0;
    const double min_val = -1.0;
    scaled_image.convertTo(output, CV_32FC3, (max_val - min_val) / 255.0,
                           min_val);
}

void DeeplabSegmentationModel::_infer(const cv::Mat& input, cv::Mat& output) {
    auto image_size =
        input.rows * input.cols * input.channels() * sizeof(float);
    input_tensor.CopyFromBytes(input.data, image_size);
    set_input("sub_7", input_tensor);
    run();
    get_output(0, output_tensor);

    constexpr auto output_tensor_size = 257 * 257 * 21;
    std::array<float, output_tensor_size> output_arr;
    output_tensor.CopyToBytes(output_arr.data(),
                              output_tensor_size * sizeof(float));
}

void DeeplabSegmentationModel::_postprocess(const cv::Mat& input,
                                            cv::Mat& output) {}
}  // namespace mukham
