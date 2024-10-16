#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class BBox {
   private:
    cv::Rect rect;
    int label;
    double confidence;

   public:
    BBox(const cv::Rect rect, const int label, const double confidence);
    cv::Rect get_rect(void) const;
    int get_label(void) const;
    double get_confidence(void) const;
};
