#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>

#include "openvino_model.hpp"
#include "result_objects.hpp"

/**
 * @brief 後処理のインタフェース
 *
 * @tparam OutputType 出力の型
 */
template <typename OutputType>
class PostprocessInterface {
   public:
    /**
     * @brief 後処理の純仮想関数
     *
     * @param model モデル
     * @return OutputType 出力
     */
    virtual OutputType postprocess(OpenVINOModel& model) = 0;
};

/**
 * @brief 検知枠とラベルを返す後処理
 *
 */
class BBoxAndLabel : public PostprocessInterface<std::vector<BBox>> {
   public:
    /**
     * @brief 検知枠とラベルを返す後処理
     *
     * @param model モデル
     * @return std::vector<BBox> BBoxのリスト
     */
    std::vector<BBox> postprocess(OpenVINOModel& model) override;
};
