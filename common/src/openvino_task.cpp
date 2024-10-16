#include "openvino_task.hpp"

template <typename Preprocess, typename Postprocess>
OpenVINOTask<Preprocess, Postprocess>::OpenVINOTask(const std::string model_path) {
    model = std::make_unique<OpenVINOModel>(model_path);
    preprocessor = Preprocess();
    postprocessor = Postprocess();
}

Detector::Detector(const std::string model_path) : OpenVINOTask(model_path) {}

cv::Mat Detector::task(const cv::Mat& image) {
    ov::Tensor input_tensor = preprocessor.preprocess(*model, image);
    model->infer(input_tensor);
    std::vector<BBox> bboxes = postprocessor.postprocess(*model);

    cv::Mat image_out = image.clone();
    const int image_width = image_out.cols;
    const int image_height = image_out.rows;
    for (BBox bbox : bboxes) {
        cv::Rect rect = bbox.get_rect();
        const int xmin = (int)(rect.x * image_width);
        const int ymin = (int)(rect.y * image_height);
        const int xmax = (int)(rect.width * image_width) + xmin;
        const int ymax = (int)(rect.height * image_height) + ymin;
        cv::rectangle(image_out, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255, 0, 0), 5);
    }
    return image_out;
}