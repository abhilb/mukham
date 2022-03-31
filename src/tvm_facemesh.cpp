
#include "tvm_facemesh.h"

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "spdlog/spdlog.h"

namespace tvm_facemesh {

bool TVM_Facemesh::Detect(const cv::Mat& input, TVM_FacemeshResult& result) {
    if (!can_execute) return false;

    // preprocessing
    cv::Mat scaled_image, preprocessed_image;
    cv::resize(input, scaled_image, cv::Size(input_width, input_height));
    scaled_image.convertTo(preprocessed_image, CV_32FC3, 1.0 / 255.0);

    auto image_size = input_width * input_height * channels * sizeof(float);
    input_tensor.CopyFromBytes(preprocessed_image.data, image_size);
    set_input("input_1", input_tensor);

    run();

    get_output(0, output_tensor_1);
    get_output(1, output_tensor_2);

    auto landmarks_tensor = std::make_unique<float[]>(nr_landmarks);
    auto face_confidence = std::make_unique<float>(1);
    output_tensor_1.CopyToBytes(landmarks_tensor.get(),
                                nr_landmarks * sizeof(float));
    output_tensor_2.CopyToBytes(face_confidence.get(), sizeof(float));

    result.face_score = *face_confidence;
    result.has_face = (*face_confidence) > 10;
    for (int idx = 0; idx < nr_landmarks; idx += 3) {
        result.mesh.push_back(
            cv::Point2d(landmarks_tensor[idx] * input.cols / input_width,
                        landmarks_tensor[idx + 1] * input.rows / input_height));
    }

    return true;
}

bool TVM_Facemesh::Detect(
    const std::vector<cv::Mat>& input,
    std::vector<tvm_facemesh::TVM_FacemeshResult>& result) {
    if (!can_execute) {
        return false;
    }

    // Preprocessing
    auto single_image_size =
        input_width * input_height * channels * sizeof(float);
    size_t batch_image_size = single_image_size * _batch_size;
    float* input_data = new float[batch_image_size];

    auto dest_data_ptr = input_data;

    std::vector<cv::Mat> batch_images;
    for (auto& image : input) {
        cv::Mat scaled_image, preprocessed_image;
        cv::resize(image, scaled_image, cv::Size(input_width, input_height));
        scaled_image.convertTo(preprocessed_image, CV_32FC4, 1.0 / 255.0, 0);
        batch_images.push_back(preprocessed_image);
    }

    cv::Mat image_batch;
    cv::vconcat(batch_images, image_batch);
    input_tensor.CopyFromBytes(image_batch.data, batch_image_size);
    set_input("input_1", input_tensor);

    // Execute the model
    run();

    // Get the output tensors
    get_output(0, output_tensor_1);
    get_output(1, output_tensor_2);

    // Postprocessing
    auto positions = std::make_unique<float[]>(nr_positions);
    auto face_flag = std::make_unique<float[]>(_batch_size * 1);
    output_tensor_1.CopyToBytes(positions.get(), nr_positions * sizeof(float));
    output_tensor_2.CopyToBytes(face_flag.get(), _batch_size * sizeof(float));

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
        result.push_back(result_item);
    }

    delete[] input_data;
    return true;
}
}  // namespace tvm_facemesh
