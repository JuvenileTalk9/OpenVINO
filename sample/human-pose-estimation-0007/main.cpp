#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "openvino_task.hpp"

namespace fs = std::filesystem;

const std::array<std::pair<int, int>, 19> SKELETON = {{{15, 13},
                                                       {13, 11},
                                                       {16, 14},
                                                       {14, 12},
                                                       {11, 12},
                                                       {5, 11},
                                                       {6, 12},
                                                       {5, 6},
                                                       {5, 7},
                                                       {6, 8},
                                                       {7, 9},
                                                       {8, 10},
                                                       {1, 2},
                                                       {0, 1},
                                                       {0, 2},
                                                       {1, 3},
                                                       {2, 4},
                                                       {3, 5},
                                                       {4, 6}}};

const std::array<cv::Scalar, 19> COLORS = {
    cv::Scalar(255, 0, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(255, 0, 255),
    cv::Scalar(170, 0, 255), cv::Scalar(255, 0, 85),  cv::Scalar(255, 0, 170),
    cv::Scalar(85, 255, 0),  cv::Scalar(255, 170, 0), cv::Scalar(0, 255, 0),
    cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 85),  cv::Scalar(170, 255, 0),
    cv::Scalar(0, 85, 255),  cv::Scalar(0, 255, 170), cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 255), cv::Scalar(85, 0, 255),  cv::Scalar(0, 170, 255),
    cv::Scalar(0, 170, 255)};

void draw_skeleton(cv::Mat& image, const std::vector<KeyPoint>& keypoints,
                   const float confidence_thr = 0.001) {
    const int image_width = image.cols;
    const int image_height = image.rows;

    for (int i = 0; i < SKELETON.size(); i++) {
        const auto& pair = SKELETON[i];
        const auto& color = COLORS[i];
        const KeyPoint keypoint1 = keypoints[pair.first];
        const KeyPoint keypoint2 = keypoints[pair.second];
        if (keypoint1.get_confidence() < confidence_thr ||
            keypoint2.get_confidence() < confidence_thr) {
            continue;
        }

        // 座標を正規化前に戻す
        const cv::Point pt1(keypoint1.get_x() * image_width, keypoint1.get_y() * image_height);
        const cv::Point pt2(keypoint2.get_x() * image_width, keypoint2.get_y() * image_height);

        // キーポイントを画像に描画
        if (pt1.x > 0 && pt1.y > 0 && pt2.x > 0 && pt2.y > 0) {
            cv::line(image, pt1, pt2, color, 2);
            cv::circle(image, pt1, 4, color, -1, cv::FILLED);
            cv::circle(image, pt2, 4, color, -1, cv::FILLED);
        }
    }
}

void process_image(DetectorBBox7& detector, PoseDetector& pose_detector, const fs::path input_path,
                   const fs::path output_path, const double confidence_thr = 0.2) {
    // 画像読み込み
    cv::Mat image = cv::imread(input_path.string());
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << input_path << std::endl;
        return;
    }

    // 可視化用の画像準備
    cv::Mat image_out = image.clone();
    const int image_width = image_out.cols;
    const int image_height = image_out.rows;

    // タスク実行
    auto bboxes = detector.task(image);

    // BBoxごとに骨格抽出
    for (BBox bbox : bboxes) {
        // 確信度が閾値以下なら無視
        if (bbox.get_confidence() < confidence_thr) {
            continue;
        }

        // 座標情報を正規化前に戻す
        cv::Rect2f rect = bbox.get_rect();
        const int xmin = static_cast<int>(rect.x * image_width);
        const int ymin = static_cast<int>(rect.y * image_height);
        const int xmax = static_cast<int>(rect.width * image_width) + xmin;
        const int ymax = static_cast<int>(rect.height * image_height) + ymin;
        cv::Rect rect_collect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
        cv::Mat cropped_image = image(rect_collect);

        // 骨格抽出
        std::vector<KeyPoint> keypoints = pose_detector.task(cropped_image);

        // 結果の描画
        cv::Mat roi(image_out, rect_collect);
        draw_skeleton(roi, keypoints);
        cropped_image.release();
    }

    // 結果の書き込み
    cv::imwrite(output_path.string(), image_out);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <detection_model_path> <pose_model_path> <input_dir> <output_dir>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    const fs::path detection_model_path = argv[1];
    const fs::path pose_model_path = argv[2];
    const fs::path input_dir = argv[3];
    const fs::path output_dir = argv[4];

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

    DetectorBBox7 detector(detection_model_path.string());
    PoseDetector pose_detector(pose_model_path.string());

    // 1画像ずつ読み込んで処理
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        try {
            if (entry.is_regular_file()) {
                fs::path input_path = entry.path();
                fs::path output_path = output_dir / input_path.filename();
                std::cout << "Processing file: " << input_path << std::endl;
                process_image(detector, pose_detector, input_path, output_path);
            }
        } catch (const ov::Exception& e) {
            std::cerr << "OpenVINO Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
