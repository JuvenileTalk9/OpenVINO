#include "openvino_model.hpp"

OpenVINOModel::OpenVINOModel(const std::string model_path, const std::string device) {
    // ファイル存在チェック
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error("No such model file: " + model_path);
    }

    // モデルの読み込み
    model = core.read_model(model_path);
    compiled_model = core.compile_model(model, device);
    infer_request = compiled_model.create_infer_request();
}

OpenVINOModel::~OpenVINOModel(void) {}

ov::Shape OpenVINOModel::get_input_shape(void) const { return compiled_model.input().get_shape(); }

ov::element::Type OpenVINOModel::get_elementtype(void) const {
    return compiled_model.input().get_element_type();
}

void OpenVINOModel::infer(ov::Tensor& input_tensor) {
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

ov::Tensor OpenVINOModel::get_output_tensor(const int index) {
    return infer_request.get_output_tensor(index);
}
