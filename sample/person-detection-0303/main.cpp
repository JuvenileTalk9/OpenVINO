#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "openvino_model.hpp"
#include "openvino_task.hpp"

namespace fs = std::filesystem;

std::vector<cv::Rect> detect(OpenVINOModel& model, const cv::Mat& image,
                             float confidence_threshold = 0.2) {
    const int height = image.rows;
    const int width = image.cols;
    const ov::Shape input_shape = model.get_input_shape();

    // プリプロセス: 画像のリサイズ、型変換、CHW変換
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(input_shape[3], input_shape[2]));
    input_image.convertTo(input_image, CV_32FC3);  // int -> float
    input_image = convert_hwc2chw(input_image);    // HWC -> CHW

    // Mat -> Tensor
    ov::Tensor input_tensor(ov::element::f32, {1, input_shape[1], input_shape[2], input_shape[3]},
                            input_image.data);

    // 推論
    model.infer(input_tensor);

    // 結果の取得
    const ov::Tensor& boxes_tensor = model.get_output_tensor(0);
    const ov::Tensor& labels_tensor = model.get_output_tensor(1);
    const int num_boxes = boxes_tensor.get_shape()[0];

    const float* boxes = boxes_tensor.data<const float>();
    const int64_t* labels = labels_tensor.data<const int64_t>();

    // 推論結果を基に、検出された矩形領域を取得
    std::vector<cv::Rect> results;
    for (int idx = 0; idx < num_boxes; idx++) {
        const float conf = boxes[idx * 5 + 4];  // 確信度の取得
        const int label = labels[idx];          // ラベルの取得

        // 確信度が閾値以下、または無効なラベルの場合は終了
        if (conf < confidence_threshold || label < 0) {
            break;
        }

        // 検出されたバウンディングボックスの座標を取得
        int xmin = static_cast<int>(boxes[idx * 5 + 0] / input_shape[3] * width);
        int ymin = static_cast<int>(boxes[idx * 5 + 1] / input_shape[2] * height);
        int xmax = static_cast<int>(boxes[idx * 5 + 2] / input_shape[3] * width);
        int ymax = static_cast<int>(boxes[idx * 5 + 3] / input_shape[2] * height);

        results.emplace_back(cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)));
    }

    return results;
}

void process_image1(OpenVINOModel& model, const fs::path input_path, const fs::path output_path) {
    // 画像読み込み
    cv::Mat image = cv::imread(input_path.string());
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << input_path << std::endl;
        return;
    }

    cv::Mat image_out = image.clone();
    auto detections = detect(model, image);
    for (auto detection : detections) {
        cv::rectangle(image_out, cv::Point(detection.x, detection.y),
                      cv::Point(detection.x + detection.width, detection.y + detection.height),
                      cv::Scalar(255, 0, 0), 5);
        cv::imwrite(output_path.string(), image_out);
    }
}

void process_image(Detector& detector, const fs::path input_path, const fs::path output_path) {
    // 画像読み込み
    cv::Mat image = cv::imread(input_path.string());
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << input_path << std::endl;
        return;
    }

    auto image_out = detector.task(image);
    cv::imwrite(output_path.string(), image_out);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <detection_model_path> <input_dir> <output_dir>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    const fs::path model_path = argv[1];
    const fs::path input_dir = argv[2];
    const fs::path output_dir = argv[3];

    // 入力ディレクトリの確認
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: Input directory does not exist: " << input_dir << std::endl;
        return EXIT_FAILURE;
    }

    // 出力ディレクトリの確認、なければ作成
    if (!fs::exists(output_dir)) {
        if (!fs::create_directories(output_dir)) {
            std::cerr << "Error: Could not create output directory: " << output_dir << std::endl;
            return EXIT_FAILURE;
        }
    }

    OpenVINOModel model(model_path.string());
    // Detector detector(model_path.string());

    // 1画像ずつ読み込んで処理
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        try {
            if (entry.is_regular_file()) {
                fs::path input_path = entry.path();
                fs::path output_path = output_dir / input_path.filename();
                std::cout << "Processing file: " << input_path << std::endl;
                process_image1(model, input_path, output_path);
                // process_image(detector, input_path, output_path);
            }
        } catch (const ov::Exception& e) {
            std::cerr << "OpenVINO Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
