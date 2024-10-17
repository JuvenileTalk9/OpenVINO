#include "postprocess.hpp"

std::vector<BBox> BBox5Label1::postprocess(OpenVINOModel& model) {
    // 結果の取得
    const ov::Shape input_shape = model.get_input_shape();
    const ov::Tensor& boxes_tensor = model.get_output_tensor(0);
    const ov::Tensor& labels_tensor = model.get_output_tensor(1);

    // プリミティブ型の配列に変換
    const float* boxes = boxes_tensor.data<const float>();
    const std::int64_t* labels = labels_tensor.data<const std::int64_t>();

    // 出力サイズの取得
    auto output_shape = boxes_tensor.get_shape();
    const std::size_t num_boxes = output_shape[0];
    const std::size_t num_data_each_box = output_shape[1];

    // 推論結果を基に、検出された矩形領域を取得
    std::vector<BBox> bboxes;
    for (int idx = 0, size = std::min(num_boxes, MAX_DETECTION); idx < size; idx++) {
        const int label = labels[idx];                                // ラベルの取得
        const float confidence = boxes[idx * num_data_each_box + 4];  // 確信度の取得

        // 無効なラベルの場合は終了
        if (label < 0) {
            break;
        }

        // 検出されたバウンディングボックスの座標を取得
        const float xmin = static_cast<float>(boxes[idx * num_data_each_box + 0] / input_shape[3]);
        const float ymin = static_cast<float>(boxes[idx * num_data_each_box + 1] / input_shape[2]);
        const float xmax = static_cast<float>(boxes[idx * num_data_each_box + 2] / input_shape[3]);
        const float ymax = static_cast<float>(boxes[idx * num_data_each_box + 3] / input_shape[2]);

        bboxes.emplace_back(cv::Rect2f(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax)), label,
                            confidence);
    }

    return bboxes;
}

std::vector<BBox> BBox7::postprocess(OpenVINOModel& model) {
    // 結果の取得
    const ov::Shape input_shape = model.get_input_shape();
    const ov::Tensor& boxes_tensor = model.get_output_tensor(0);

    // プリミティブ型の配列に変換
    const float* boxes = boxes_tensor.data<const float>();

    // 出力サイズの取得
    auto output_shape = boxes_tensor.get_shape();
    const std::size_t num_boxes = output_shape[2];
    const std::size_t num_data_each_box = output_shape[3];

    // 推論結果を基に、検出された矩形領域を取得
    std::vector<BBox> bboxes;
    for (int idx = 0, size = std::min(num_boxes, MAX_DETECTION); idx < size; idx++) {
        const int image_id = boxes[idx * num_data_each_box + 0];      // 画像IDの取得
        const int label = boxes[idx * num_data_each_box + 1];         // ラベルの取得
        const float confidence = boxes[idx * num_data_each_box + 2];  // 確信度の取得

        // 画像IDが負値になれば終了
        if (image_id < 0) {
            break;
        }

        // 検出されたバウンディングボックスの座標を取得
        const float xmin = static_cast<float>(boxes[idx * num_data_each_box + 3]);
        const float ymin = static_cast<float>(boxes[idx * num_data_each_box + 4]);
        const float xmax = static_cast<float>(boxes[idx * num_data_each_box + 5]);
        const float ymax = static_cast<float>(boxes[idx * num_data_each_box + 6]);

        bboxes.emplace_back(cv::Rect2f(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax)), label,
                            confidence);
    }

    return bboxes;
}

std::vector<KeyPoint> KeyPoints::postprocess(OpenVINOModel& model) {
    // 結果を取得
    const ov::Tensor& heatmaps_tensor = model.get_output_tensor(0);
    const float* heatmaps = reinterpret_cast<float*>(heatmaps_tensor.data());
    const ov::Shape output_shape = heatmaps_tensor.get_shape();

    // キーポイントを検出
    std::vector<KeyPoint> keypoints;
    for (int i = 0; i < output_shape[1]; i++) {
        const int max_index =
            std::max_element(heatmaps + i * output_shape[2] * output_shape[3],
                             heatmaps + (i + 1) * output_shape[2] * output_shape[3]) -
            (heatmaps + i * output_shape[2] * output_shape[3]);
        const float x = (float)(max_index % output_shape[2]) / output_shape[3];
        const float y = (float)(max_index / output_shape[2]) / output_shape[2];
        const float confidence = heatmaps[max_index + i * output_shape[2] * output_shape[3]];

        keypoints.emplace_back(x, y, confidence);
    }

    return keypoints;
}
