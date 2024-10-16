#include "openvino_task.hpp"

template <typename Preprocess, typename Postprocess>
OpenVINOTask<Preprocess, Postprocess>::OpenVINOTask(const std::string model_path) {
    model = std::make_unique<OpenVINOModel>(model_path);
    preprocessor = Preprocess();
    postprocessor = Postprocess();
}

Detector::Detector(const std::string model_path) : OpenVINOTask(model_path) {}

std::vector<BBox> Detector::task(const cv::Mat& image) {
    ov::Tensor input_tensor = preprocessor.preprocess(*model, image);
    model->infer(input_tensor);
    std::vector<BBox> bboxes = postprocessor.postprocess(*model);
    return bboxes;
}
