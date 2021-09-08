#include "tvm_blazeface.h"

#include <algorithm>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "spdlog/spdlog.h"

namespace tvm_blazeface {

void PreprocessImage(const cv::Mat& input_image, cv::Mat& output_image,
                     const cv::Size& output_size, double min_val,
                     double max_val) {
    cv::Mat scaled_image;
    cv::resize(input_image, scaled_image, output_size);
    scaled_image.convertTo(output_image, CV_32FC3, (max_val - min_val) / 127.5,
                           min_val);
}

std::vector<cv::Rect2d> TVM_Blazeface::DetectFace(const cv::Mat& input_image) {
    // preprocessing
    cv::Mat preprocessed_image;
    PreprocessImage(input_image, preprocessed_image,
                    cv::Size(anchor_options.input_size_width,
                             anchor_options.input_size_height),
                    -1.0, 1.0);

    // Set the input
    auto single_image_size =
        preprocessed_image.total() * preprocessed_image.elemSize();
    input_tensor.CopyFromBytes(preprocessed_image.data, single_image_size);
    set_input("input", input_tensor);

    // Execute the model
    run();

    // Get the output tensors
    get_output(0, output_tensor_1);
    get_output(1, output_tensor_2);

    // Get the raw boxes
    auto output_tensor_1_size =
        box_options.num_boxes * box_options.num_coords * batch_size;

    auto raw_boxes = std::make_unique<float[]>(output_tensor_1_size);

    output_tensor_1.CopyToBytes(
        raw_boxes.get(),
        output_tensor_1_size * output_tensor_1.DataType().bytes());

    // Get the raw scores
    auto output_tensor_2_size = box_options.num_boxes * batch_size;

    auto raw_scores = std::make_unique<float[]>(output_tensor_2_size);

    output_tensor_2.CopyToBytes(
        raw_scores.get(),
        output_tensor_2_size * output_tensor_2.DataType().bytes());

    // Convert to boxes
    std::vector<cv::Rect2d> boxes;
    _decode_boxes(std::move(raw_boxes), std::move(raw_scores), boxes);

    return boxes;
}

void TVM_Blazeface::_get_anchor_boxes() {
    int layer_id = 0, last_same_stride_layer = 0, repeats = 0;

    while (layer_id < anchor_options.num_layers) {
        last_same_stride_layer = layer_id;
        repeats = 0;
        while (last_same_stride_layer < anchor_options.num_layers &&
               anchor_options.strides[last_same_stride_layer] ==
                   anchor_options.strides[layer_id]) {
            last_same_stride_layer += 1;
            repeats += 2;
        }

        auto stride = anchor_options.strides[layer_id];
        int feature_map_height = anchor_options.input_size_height / stride;
        int feature_map_width = anchor_options.input_size_width / stride;

        for (int y = 0; y < feature_map_height; y++) {
            double y_center =
                (y + anchor_options.anchor_offset_y) / feature_map_height;
            for (int x = 0; x < feature_map_width; x++) {
                double x_center =
                    (x + anchor_options.anchor_offset_x) / feature_map_width;
                for (int r = 0; r < repeats; r++) {
                    anchors.push_back(std::make_pair(x_center, y_center));
                }
            }
        }

        layer_id = last_same_stride_layer;
    }
}

void TVM_Blazeface::_decode_boxes(std::unique_ptr<float[]> raw_boxes,
                                  std::unique_ptr<float[]> raw_scores,
                                  std::vector<cv::Rect2d>& boxes) {
    std::vector<std::pair<double, cv::Rect2d>> detections;
    for (int i = 0; i < num_boxes; ++i) {
        auto score = raw_scores[i];
        if (box_options.sigmoid_score) {
            score =
                std::clamp<double>(score, -box_options.score_clipping_thresh,
                                   box_options.score_clipping_thresh);
            score = 1 / (1 + std::exp((double)(-score)));
        }

        if (score < box_options.min_score_thresh) continue;

        const int box_offset =
            i * box_options.num_coords + box_options.box_coord_offset;
        float y_center = raw_boxes[box_offset];
        float x_center = raw_boxes[box_offset + 1];
        float h = raw_boxes[box_offset + 2];
        float w = raw_boxes[box_offset + 3];

        x_center = (x_center / box_options.x_scale) + anchors[i].first;
        y_center = (y_center / box_options.y_scale) + anchors[i].second;
        w = w / box_options.x_scale;
        h = h / box_options.y_scale;

        if (box_options.reverse_output_order) {
            std::swap(x_center, y_center);
            std::swap(h, w);
        }

        if (h < 0 || w < 0) continue;

        const float left = x_center - w / 2.f;
        const float top = y_center - h / 2.f;
        detections.push_back(
            std::make_pair(score, cv::Rect2d(left, top, w, h)));
    }

    std::sort(detections.begin(), detections.end(),
              [](auto a, auto b) { return a.first < b.first; });

    _nms(detections, boxes);
}

void TVM_Blazeface::_nms(
    const std::vector<std::pair<double, cv::Rect2d>> detections,
    std::vector<cv::Rect2d>& output) {
    const double min_supression_threshold = 0.3;

    std::vector<cv::Rect2d> kept_boxes;
    bool supressed = false;

    for (const auto& [score, box] : detections) {
        if (score < box_options.min_score_thresh) {
            break;
        }

        supressed = false;
        for (const auto& kept : kept_boxes) {
            auto similarity = _overlap_similarity(kept, box);
            if (similarity > min_supression_threshold) {
                supressed = true;
                break;
            }
        }

        if (!supressed) {
            kept_boxes.push_back(box);
            auto scaled_box = cv::Rect2d(box.x * box_options.x_scale,
                                         box.y * box_options.y_scale,
                                         box.width * box_options.x_scale,
                                         box.height * box_options.y_scale);
            output.push_back(scaled_box);
        }
    }
    spdlog::info("Number of boxes: {}", output.size());
}

double TVM_Blazeface::_overlap_similarity(const cv::Rect2d& box1,
                                          const cv::Rect2d& box2) {
    auto intersection = _get_intesection(box1, box2);
    if (intersection.empty()) return 0.0;

    auto numerator = intersection.area();
    auto denominator = box1.area() + box2.area() - intersection.area();
    if (denominator <= 0) return 0.0;
    return numerator / denominator;
}

cv::Rect2d TVM_Blazeface::_get_intesection(const cv::Rect2d& box1,
                                           const cv::Rect2d& box2) {
    const auto xmin1 = box1.x;
    const auto ymin1 = box1.y;
    const auto xmax1 = box1.x + box1.width;
    const auto ymax1 = box1.y + box1.height;

    const auto xmin2 = box2.x;
    const auto ymin2 = box2.y;
    const auto xmax2 = box2.x + box2.width;
    const auto ymax2 = box2.y + box2.height;

    auto xmin = (std::max)(xmin1, xmin2);
    auto ymin = (std::max)(ymin1, ymin2);
    auto xmax = (std::min)(xmax1, xmax2);
    auto ymax = (std::min)(ymax1, ymax2);

    if ((xmin < xmax) && (ymin < ymax)) {
        return cv::Rect2d(xmin, ymin, (xmax - xmin), (ymax - ymin));
    }
    return cv::Rect2d();
}
}  // namespace tvm_blazeface
