#include "postprocess.hpp"

std::vector<BBox> BBoxAndLabel::postprocess(OpenVINOModel& model) {
    // 結果の取得
    const ov::Shape input_shape = model.get_input_shape();
    const ov::Tensor& boxes_tensor = model.get_output_tensor(0);
    const ov::Tensor& labels_tensor = model.get_output_tensor(1);
    const int num_boxes = boxes_tensor.get_shape()[0];

    // プリミティブ型の配列に変換
    const float* boxes = boxes_tensor.data<const float>();
    const int64_t* labels = labels_tensor.data<const int64_t>();

    // 推論結果を基に、検出された矩形領域を取得
    std::vector<BBox> bboxes;
    for (int idx = 0; idx < num_boxes; idx++) {
        const int label = labels[idx];                // ラベルの取得
        const float confidence = boxes[idx * 5 + 4];  // 確信度の取得

        // 無効なラベルの場合は終了
        if (label < 0) {
            break;
        }

        // 検出されたバウンディングボックスの座標を取得
        const int xmin = static_cast<int>(boxes[idx * 5 + 0] / input_shape[3]);
        const int ymin = static_cast<int>(boxes[idx * 5 + 1] / input_shape[2]);
        const int xmax = static_cast<int>(boxes[idx * 5 + 2] / input_shape[3]);
        const int ymax = static_cast<int>(boxes[idx * 5 + 3] / input_shape[2]);

        bboxes.emplace_back(cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), label,
                            confidence);
    }

    return bboxes;
}