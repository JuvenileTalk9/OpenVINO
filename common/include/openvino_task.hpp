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
 * @brief 物体検知タスク（BBox5Label1）
 *
 */
class DetectorBBox5Label1 : public OpenVINOTask<FloatCHW, BBox5Label1> {
   public:
    /**
     * @brief モデルを読み込む
     *
     * @param model_path モデルファイルのパス
     */
    DetectorBBox5Label1(const std::string model_path);

    /**
     * @brief 物体検知タスクを実行する
     *
     * @param image 入力画像
     * @return std::vector<BBox> 検知結果のベクトル
     */
    std::vector<BBox> task(const cv::Mat& image);
};

/**
 * @brief 物体検知タスク（BBox7）
 *
 */
class DetectorBBox7 : public OpenVINOTask<FloatCHW, BBox7> {
   public:
    /**
     * @brief モデルを読み込む
     *
     * @param model_path モデルファイルのパス
     */
    DetectorBBox7(const std::string model_path);

    /**
     * @brief 物体検知タスクを実行する
     *
     * @param image 入力画像
     * @return std::vector<BBox> 検知結果のベクトル
     */
    std::vector<BBox> task(const cv::Mat& image);
};

/**
 * @brief 骨格検出タスク
 *
 */
class PoseDetector : public OpenVINOTask<FloatCHW, KeyPoints> {
   public:
    /**
     * @brief モデルを読み込む
     *
     * @param model_path モデルファイルのパス
     */
    PoseDetector(const std::string model_path);

    /**
     * @brief 骨格検出タスクを実行する
     *
     * @param image 入力画像
     * @return std::vector<KeyPoint> キーポイントのベクトル
     */
    std::vector<KeyPoint> task(const cv::Mat& image);
};
