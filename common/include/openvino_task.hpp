#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "openvino_model.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"

template <typename Preprocess, typename Postprocess>
class OpenVINOTask {
   protected:
    std::unique_ptr<OpenVINOModel> model = nullptr;
    Preprocess preprocessor;
    Postprocess postprocessor;

   public:
    OpenVINOTask(const std::string model_path);
};

class Detector : public OpenVINOTask<FloatCHW, BBoxAndLabel> {
   public:
    Detector(const std::string model_path);
    cv::Mat task(const cv::Mat& image);
};
