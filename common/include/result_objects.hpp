#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

/**
 * @brief バウンディングボックスのオブジェクトクラス
 *
 */
class BBox {
   private:
    cv::Rect2f rect;
    int label;
    double confidence;

   public:
    /**
     * @brief コンストラクタ
     *
     * @param rect 座標
     * @param label ラベル
     * @param confidence 確信度
     */
    BBox(const cv::Rect2f rect, const int label, const double confidence);

    /**
     * @brief 座標を返す
     *
     * @return cv::Rect2f 座標
     */
    cv::Rect2f get_rect(void) const;

    /**
     * @brief ラベルを返す
     *
     * @return int ラベル
     */
    int get_label(void) const;

    /**
     * @brief 確信度を返す
     *
     * @return double 確信度
     */
    double get_confidence(void) const;
};
