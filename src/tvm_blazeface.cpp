#include "tvm_blazeface.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "spdlog/spdlog.h"

namespace tvm_blazeface {

void PreprocessImage(const cv::Mat& input_image, const cv::Size& output_size,
                     double min_val, double max_val, cv::Mat& output_image,
                     int& padx, int& pady) {
    padx = input_image.rows > input_image.cols
               ? (input_image.rows - input_image.cols) >> 1
               : 0;
    pady = input_image.cols > input_image.rows
               ? (input_image.cols - input_image.rows) >> 1
               : 0;

    cv::Mat scaled_image;
    cv::Mat input_image_with_border;
    cv::copyMakeBorder(input_image, input_image_with_border, pady, pady, padx,
                       padx, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::resize(input_image_with_border, scaled_image, output_size,
               cv::INTER_AREA);
    scaled_image.convertTo(output_image, CV_32FC3, (max_val - min_val) / 255.0,
                           min_val);
}

std::vector<Detection> TVM_Blazeface::DetectFace(const cv::Mat& input_image) {
    // preprocessing
    cv::Mat preprocessed_image;
    auto expected_input_size = cv::Size(anchor_options.input_size_width,
                                        anchor_options.input_size_height);

    int padx, pady;
    PreprocessImage(input_image, expected_input_size, -1.0, 1.0,
                    preprocessed_image, padx, pady);

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
    std::vector<Detection> detections;
    _decode_boxes(std::move(raw_boxes), std::move(raw_scores), detections);

    auto scale_factor = (std::max)(input_image.rows, input_image.cols);
    for (auto& d : detections) {
        auto& box = d.bounding_box;
        box.x = (box.x * scale_factor) - padx;
        box.y = (box.y * scale_factor) - pady;
        box.width *= scale_factor;
        box.height *= scale_factor;

        for (auto& kp : d.key_points) {
            kp.x = (kp.x * scale_factor) - padx;
            kp.y = (kp.y * scale_factor) - pady;
        }
    }

    return detections;
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
                                  std::vector<Detection>& detections) {
    DetectionsVec all_detections;
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
        if (box_options.reverse_output_order) {
            std::swap(x_center, y_center);
            std::swap(h, w);
        }

        x_center = (x_center / box_options.x_scale) + anchors[i].first;
        y_center = (y_center / box_options.y_scale) + anchors[i].second;
        w = w / box_options.x_scale;
        h = h / box_options.y_scale;

        std::array<cv::Point2d, 6> key_points;
        for (int kidx = 0; kidx < box_options.num_keypoints; kidx++) {
            auto keypoint_offset = i * box_options.num_coords +
                                   box_options.keypoint_coord_offset +
                                   kidx * box_options.num_values_per_keypoint;
            auto keypoint_y = raw_boxes[keypoint_offset];
            auto keypoint_x = raw_boxes[keypoint_offset + 1];
            if (box_options.reverse_output_order) {
                std::swap(keypoint_y, keypoint_x);
            }

            key_points[kidx] = cv::Point2d(
                keypoint_x / box_options.x_scale + anchors[i].first,
                keypoint_y / box_options.y_scale + anchors[i].second);
        }

        if (h < 0 || w < 0) continue;

        const float left = x_center - w / 2.f;
        const float top = y_center - h / 2.f;
        all_detections.push_back({.score = score,
                                  .bounding_box = cv::Rect2d(left, top, w, h),
                                  .key_points = std::move(key_points)});
    }

    _weighted_nms(all_detections, detections);
}

void TVM_Blazeface::_make_indexed_scores(const DetectionsVec& detections,
                                         IndexedScoresVec& idx_scores) {
    int index = 0;
    for (const auto& detection : detections) {
        idx_scores.push_back(std::make_pair(index, (double)(detection.score)));
        index++;
    }
}

void TVM_Blazeface::_weighted_nms(const std::vector<Detection> detections,
                                  std::vector<Detection>& output) {
    std::vector<IndexedScore> indexed_scores;
    _make_indexed_scores(detections, indexed_scores);

    // sort the index scores
    std::sort(indexed_scores.begin(), indexed_scores.end(),
              [](auto a, auto b) { return a.second > b.second; });

    IndexedScoresVec remaining;
    IndexedScoresVec candidates;
    const double min_supression_threshold = 0.3;

    while (!indexed_scores.empty()) {
        const int original_indexed_scores_size = indexed_scores.size();
        const Detection detection = detections[indexed_scores.front().first];

        if (detection.score < box_options.min_score_thresh) break;

        remaining.clear();
        candidates.clear();

        auto detection_box = detection.bounding_box;
        for (const auto& idx_score : indexed_scores) {
            auto other_bbox = detections[idx_score.first].bounding_box;
            auto similarity = _overlap_similarity(detection_box, other_bbox);

            if (similarity > min_supression_threshold)
                candidates.push_back(idx_score);
            else
                remaining.push_back(idx_score);
        }

        auto weighted_detection = detection;
        if (!candidates.empty()) {
            double wxmin = 0.0;
            double wxmax = 0.0;
            double wymin = 0.0;
            double wymax = 0.0;
            double total_score = 0.0;

            for (auto& candidate : candidates) {
                total_score += candidate.second;
                auto bbox = detections[candidate.first].bounding_box;
                wxmin += (bbox.x * candidate.second);
                wymin += (bbox.y * candidate.second);
                wxmax += ((bbox.width + bbox.x) * candidate.second);
                wymax += ((bbox.height + bbox.y) * candidate.second);
            }

            if (total_score > 0) {
                weighted_detection.bounding_box.x = wxmin / total_score;
                weighted_detection.bounding_box.y = wymin / total_score;
                weighted_detection.bounding_box.width =
                    (wxmax / total_score) - weighted_detection.bounding_box.x;
                weighted_detection.bounding_box.height =
                    (wymax / total_score) - weighted_detection.bounding_box.y;
            }
        }

        output.push_back(weighted_detection);

        if (original_indexed_scores_size == remaining.size())
            break;
        else
            indexed_scores = std::move(remaining);
    }
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
            output.push_back(box);
        }
    }
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
