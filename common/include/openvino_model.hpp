#pragma once

#include <filesystem>
#include <openvino/openvino.hpp>

/**
 * @brief OpenVINOのモデルを管理する
 *
 */
class OpenVINOModel {
   protected:
    ov::Core core;
    std::shared_ptr<ov::Model> model = nullptr;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

   public:
    /**
     * @brief モデルを読み込んで推論可能な状態にする
     *
     * @param model_path モデルのパス
     */
    OpenVINOModel(const std::string model_path);

    /**
     * @brief 読み込んだモデルを解放する
     *
     */
    ~OpenVINOModel(void);

    /**
     * @brief モデルの入力サイズを返す
     *
     * @return ov::Shape モデルの入力サイズ
     */
    ov::Shape get_input_shape(void) const;

    /**
     * @brief 推論を実行する
     *
     * @param input_tensor 入力テンソル
     */
    void infer(ov::Tensor& input_tensor);

    /**
     * @brief 推論結果を取得する
     *
     * @param index 取得する属性インデックス
     * @return ov::Tensor 出力テンソル
     */
    ov::Tensor get_output_tensor(const int index);
};
