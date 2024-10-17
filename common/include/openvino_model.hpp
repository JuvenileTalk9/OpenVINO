#pragma once

#include <exception>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <string>

/**
 * @brief OpenVINOのモデルを管理する
 *
 */
class OpenVINOModel {
   private:
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
    OpenVINOModel(const std::string model_path, const std::string device = "CPU");

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
     * @brief 入力の型情報を返す
     *
     * @return ov::element::Type 入力の型情報
     */
    ov::element::Type get_elementtype(void) const;

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
