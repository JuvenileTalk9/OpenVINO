#include "openvino_task.hpp"

template <typename Preprocess, typename Postprocess>
OpenVINOTask<Preprocess, Postprocess>::OpenVINOTask(const std::string model_path) {
    model = std::make_unique<OpenVINOModel>(model_path);
    preprocessor = Preprocess();
    postprocessor = Postprocess();
}

DetectorBBox5Label1::DetectorBBox5Label1(const std::string model_path) : OpenVINOTask(model_path) {}

std::vector<BBox> DetectorBBox5Label1::task(const cv::Mat& image) {
    ov::Tensor input_tensor = preprocessor.preprocess(*model, image);
    model->infer(input_tensor);
    std::vector<BBox> bboxes = postprocessor.postprocess(*model);
    return bboxes;
}

DetectorBBox7::DetectorBBox7(const std::string model_path) : OpenVINOTask(model_path) {}

std::vector<BBox> DetectorBBox7::task(const cv::Mat& image) {
    ov::Tensor input_tensor = preprocessor.preprocess(*model, image);
    model->infer(input_tensor);
    std::vector<BBox> bboxes = postprocessor.postprocess(*model);
    return bboxes;
}

PoseDetector::PoseDetector(const std::string model_path) : OpenVINOTask(model_path) {}

std::vector<KeyPoint> PoseDetector::task(const cv::Mat& image) {
    ov::Tensor input_tensor = preprocessor.preprocess(*model, image);
    model->infer(input_tensor);
    std::vector<KeyPoint> bboxes = postprocessor.postprocess(*model);
    return bboxes;
}
