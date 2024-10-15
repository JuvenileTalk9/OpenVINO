#include "openvino_model.hpp"

OpenVINOModel::OpenVINOModel(const std::string model_path) {
    // ファイル存在チェック
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error("No such model file: " + model_path);
    }

    // モデルの読み込み
    model = core.read_model(model_path);
    compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
}

OpenVINOModel::~OpenVINOModel(void) {}

ov::Shape OpenVINOModel::get_input_shape(void) const { return compiled_model.input().get_shape(); }

void OpenVINOModel::infer(ov::Tensor& input_tensor) {
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

ov::Tensor OpenVINOModel::get_output_tensor(const int index) { return infer_request.get_output_tensor(index); }
