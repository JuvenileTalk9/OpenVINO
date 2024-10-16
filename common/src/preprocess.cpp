#include "preprocess.hpp"

ov::Tensor FloatCHW::preprocess(const OpenVINOModel& model, const cv::Mat& image) {
    cv::Mat input_image;
    const ov::Shape input_shape = model.get_input_shape();

    // resize
    cv::resize(image, input_image, cv::Size(input_shape[3], input_shape[2]));
    // int -> float
    input_image.convertTo(input_image, CV_32FC3);
    // HWC -> CHW
    input_image = convert_hwc2chw(input_image);

    // Mat -> Tensor
    allocated_image = input_image;
    ov::Tensor input_tensor(ov::element::f32, {1, input_shape[1], input_shape[2], input_shape[3]},
                            allocated_image.data);

    return input_tensor;
}

cv::Mat convert_hwc2chw(const cv::Mat& image) {
    // チャンネル数を確認（3チャンネルが前提: BGR/RGB形式）
    const int channels = image.channels();
    const int height = image.rows;
    const int width = image.cols;

    // HWC形式のMatから各チャンネルに分離（BGR -> B, G, R）
    std::vector<cv::Mat> chw_channels;
    cv::split(image, chw_channels);  // 画像をチャンネルごとに分割

    // CHW形式に変換（チャンネル->行列を1次元化して結合）
    std::vector<float> chw_data;
    for (int c = 0; c < channels; ++c) {
        std::vector<float> channel_data;
        chw_channels[c].reshape(1, 1).convertTo(channel_data, CV_32F);  // フロートに変換し1次元化
        chw_data.insert(chw_data.end(), channel_data.begin(), channel_data.end());
    }

    // CHW形式のフロート配列をMatに再構成
    cv::Mat chw_image = cv::Mat(chw_data, true).reshape(1, {channels, height, width});

    return chw_image;
}
