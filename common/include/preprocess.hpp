#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "openvino_model.hpp"

class PreprocessInterface {
   public:
    virtual ov::Tensor preprocess(const OpenVINOModel& model, const cv::Mat& image) = 0;
};

class FloatCHW : public PreprocessInterface {
   public:
    ov::Tensor preprocess(const OpenVINOModel& model, const cv::Mat& image) override;
};

cv::Mat convert_hwc2chw(const cv::Mat& image);
