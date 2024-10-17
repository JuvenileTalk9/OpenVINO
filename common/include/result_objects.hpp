#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

/**
 * @brief バウンディングボックスのオブジェクトクラス
 *
 */
class BBox {
   private:
    const cv::Rect2f rect;
    const int label;
    const float confidence;

   public:
    /**
     * @brief コンストラクタ
     *
     * @param rect 座標
     * @param label ラベル
     * @param confidence 確信度
     */
    BBox(const cv::Rect2f rect, const int label, const float confidence);

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
     * @return float 確信度
     */
    float get_confidence(void) const;
};

/**
 * @brief 骨格情報のオブジェクトクラス
 *
 */
class KeyPoint {
   private:
    const float x;
    const float y;
    const float confidence;

   public:
    /**
     * @brief コンストラクタ
     *
     * @param x x座標
     * @param y y座標
     * @param confidence 確信度
     */
    KeyPoint(const float x, const float y, const float confidence);

    /**
     * @brief x座標を返す
     *
     * @return float x座標
     */
    float get_x(void) const;

    /**
     * @brief y座標を返す
     *
     * @return float y座標
     */
    float get_y(void) const;

    /**
     * @brief 確信度を返す
     *
     * @return float 確信度
     */
    float get_confidence(void) const;
};
