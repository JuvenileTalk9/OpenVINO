#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "openvino_model.hpp"

/**
 * @brief 前処理のインタフェース
 *
 */
class PreprocessInterface {
   public:
    /**
     * @brief 前処理の純仮想関数
     *
     * @param model モデル
     * @param image 入力画像
     * @return ov::Tensor モデルへ入力するテンソル
     */
    virtual ov::Tensor preprocess(const OpenVINOModel& model, const cv::Mat& image) = 0;
};

/**
 * @brief float型のCHW配列へ変換する前処理
 *
 */
class FloatCHW : public PreprocessInterface {
   private:
    /**
     * @brief Tensorが示す画像のメモリが解放されないように保持しておく変数
     *
     */
    cv::Mat allocated_image;

   public:
    /**
     * @brief float型のCHW配列へ変換する前処理
     *
     * @param model モデル
     * @param image 入力画像
     * @return ov::Tensor モデルへ入力するテンソル
     */
    ov::Tensor preprocess(const OpenVINOModel& model, const cv::Mat& image) override;
};

/**
 * @brief 画像をHWC形式からCHW形式へ変換する
 *
 * @param image 変換前の画像
 * @return cv::Mat 変換後の画像
 */
cv::Mat convert_hwc2chw(const cv::Mat& image);
