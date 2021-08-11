
#include "tvm_facemesh.h"

#include <chrono>
#include <memory>
#include <opencv2/core/mat.hpp>

#include "spdlog/spdlog.h"

namespace tvm_facemesh {
bool TVM_Facemesh::Detect(
    const std::vector<cv::Mat>& input,
    std::vector<tvm_facemesh::TVM_FacemeshResult>& result) {
    if (!can_execute) {
        return false;
    }

    auto start = std::chrono::steady_clock::now();
    // Preprocessing
    auto single_image_size =
        input_width * input_height * channels * sizeof(float);
    size_t batch_image_size = single_image_size * _batch_size;
    float* input_data = new float[batch_image_size];

    auto dest_data_ptr = input_data;
    for (auto& input_image : input) {
        cv::Mat scaled_image;
        cv::Mat preprocessed_image;
        cv::resize(input_image, scaled_image,
                   cv::Size(input_width, input_height));
        scaled_image.convertTo(preprocessed_image, CV_32FC3, 1.0 / 255.0, 0);
        std::memcpy(input_data, preprocessed_image.data, single_image_size);
        dest_data_ptr += (preprocessed_image.rows * preprocessed_image.cols);
        scaled_image.release();
        preprocessed_image.release();
    }

    // Copy the image to the input tensor of the neural network
    input_tensor.CopyFromBytes(static_cast<float*>(input_data),
                               batch_image_size);
    set_input("input_1", input_tensor);

    // Execute the model
    run();

    // Get the output tensors
    get_output(0, output_tensor_1);
    get_output(1, output_tensor_2);

    // Postprocessing
    float positions[nr_positions];
    float face_flag[_batch_size * 1];

    output_tensor_1.CopyToBytes(&(positions[0]), nr_positions * sizeof(float));
    output_tensor_2.CopyToBytes(&face_flag, _batch_size * sizeof(float));

    auto end = std::chrono::steady_clock::now();
    auto processing_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    for (int batch_idx = 0; batch_idx < _batch_size; ++batch_idx) {
        tvm_facemesh::TVM_FacemeshResult result_item;

        auto start_idx = batch_idx * nr_landmarks;
        auto stop_idx = start_idx + nr_landmarks;
        for (int idx = start_idx; idx < stop_idx; idx += 3) {
            float x = positions[idx] * input[batch_idx].cols / input_width;
            float y = positions[idx + 1] * input[batch_idx].rows / input_height;
            result_item.mesh.push_back(cv::Point(x, y));
        }
        result_item.has_face = face_flag[batch_idx] > 10.0;
        result_item.face_score = face_flag[batch_idx];
        result_item.processing_time = processing_time / _batch_size;
        result.push_back(result_item);
    }

    delete[] input_data;
    return true;
}
}  // namespace tvm_facemesh