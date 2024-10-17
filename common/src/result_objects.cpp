#include "result_objects.hpp"

BBox::BBox(const cv::Rect2f rect, const int label, const float confidence)
    : rect(rect), label(label), confidence(confidence) {}

cv::Rect2f BBox::get_rect(void) const { return rect; };

int BBox::get_label(void) const { return label; }

float BBox::get_confidence(void) const { return confidence; }

KeyPoint::KeyPoint(const float x, const float y, const float confidence)
    : x(x), y(y), confidence(confidence) {}

float KeyPoint::get_x(void) const { return x; }

float KeyPoint::get_y(void) const { return y; }

float KeyPoint::get_confidence(void) const { return confidence; }
