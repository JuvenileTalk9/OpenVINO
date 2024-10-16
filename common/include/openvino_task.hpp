#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "openvino_model.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"

/**
 * @brief タスクの規程クラス
 *
 * @tparam Preprocess 前処理
 * @tparam Postprocess 後処理
 */
template <typename Preprocess, typename Postprocess>
class OpenVINOTask {
   protected:
    std::unique_ptr<OpenVINOModel> model = nullptr;
    Preprocess preprocessor;
    Postprocess postprocessor;

   public:
    /**
     * @brief モデルを読み込む
     *
     * @param model_path モデルファイルのパス
     */
    OpenVINOTask(const std::string model_path);
};

/**
 * @brief 物体検知タスク
 *
 */
class Detector : public OpenVINOTask<FloatCHW, BBoxAndLabel> {
   public:
    /**
     * @brief モデルを読み込む
     *
     * @param model_path モデルファイルのパス
     */
    Detector(const std::string model_path);

    /**
     * @brief 物体検知タスクを実行する
     *
     * @param image 入力画像
     * @return std::vector<BBox> 検知結果のベクトル
     */
    std::vector<BBox> task(const cv::Mat& image);
};
