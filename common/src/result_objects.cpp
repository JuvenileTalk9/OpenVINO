#include "result_objects.hpp"

BBox::BBox(const cv::Rect2f rect, const int label, const double confidence)
    : rect(rect), label(label), confidence(confidence) {}

cv::Rect2f BBox::get_rect(void) const { return rect; };

int BBox::get_label(void) const { return label; }

double BBox::get_confidence(void) const { return confidence; }
