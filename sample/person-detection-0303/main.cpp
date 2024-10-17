#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "openvino_task.hpp"

namespace fs = std::filesystem;

void process_image(DetectorBBox5Label1& detector, const fs::path input_path,
                   const fs::path output_path, const double confidence_thr = 0.2) {
    // 画像読み込み
    cv::Mat image = cv::imread(input_path.string());
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << input_path << std::endl;
        return;
    }
    // タスク実行
    auto bboxes = detector.task(image);

    // 結果の可視化
    cv::Mat image_out = image.clone();
    const int image_width = image_out.cols;
    const int image_height = image_out.rows;
    for (BBox bbox : bboxes) {
        // 確信度が閾値以下なら無視
        if (bbox.get_confidence() < confidence_thr) {
            continue;
        }
        cv::Rect2f rect = bbox.get_rect();
        const int xmin = static_cast<int>(rect.x * image_width);
        const int ymin = static_cast<int>(rect.y * image_height);
        const int xmax = static_cast<int>(rect.width * image_width) + xmin;
        const int ymax = static_cast<int>(rect.height * image_height) + ymin;
        cv::rectangle(image_out, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255, 0, 0), 5);
    }

    // 結果の書き込み
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

    DetectorBBox5Label1 detector(model_path.string());

    // 1画像ずつ読み込んで処理
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        try {
            if (entry.is_regular_file()) {
                fs::path input_path = entry.path();
                fs::path output_path = output_dir / input_path.filename();
                std::cout << "Processing file: " << input_path << std::endl;
                process_image(detector, input_path, output_path);
            }
        } catch (const ov::Exception& e) {
            std::cerr << "OpenVINO Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
