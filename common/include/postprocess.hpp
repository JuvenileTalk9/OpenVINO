#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>

#include "openvino_model.hpp"
#include "result_objects.hpp"

template <typename OutputType>
class PostprocessInterface {
   public:
    virtual OutputType postprocess(OpenVINOModel& model) = 0;
};

class BBoxAndLabel : public PostprocessInterface<std::vector<BBox>> {
   public:
    std::vector<BBox> postprocess(OpenVINOModel& model) override;
};
