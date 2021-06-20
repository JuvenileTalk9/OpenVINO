#include <iostream>
#include <string>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <inference_engine.hpp>


std::array<std::string, 7> COLOR_LABEL{"white", "gray", "yellow", "red", "green", "blue", "black"};
std::array<std::string, 4> TYPE_LABEL{"car", "van", "truck", "bus"};


int main(int argc, char** argv) {

    /* Check arguments */
    if (argc != 4) {
        std::cerr << "Usage: ./vehicle_recognition <model.xml> <model.bin> <image>" << std::endl;
        return -1;
    }

    std::string model_xml(argv[1]);
    std::string model_bin(argv[2]);
    std::string image_path(argv[3]);

    /* Load IR model and configure batch size */
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(model_xml, model_bin);
    network.setBatchSize(1);

    /* Configure input blobs */
    InferenceEngine::InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_info->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
    input_info->setLayout(InferenceEngine::Layout::NHWC);
    input_info->setPrecision(InferenceEngine::Precision::U8);
    
    /* Configure output blobs */
    InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
    output_info->setPrecision(InferenceEngine::Precision::FP32);

    /* Loading model to device */
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
    InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();

    /* Load input image */
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Faild to load image file: " << image_path << std::endl;
        return -1;
    }

    /* Set input image to input blob */
    const unsigned int channels = image.channels();
    const unsigned int height = image.size().height;
    const unsigned int width = image.size().width;
    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                    {1, channels, height, width},
                                    InferenceEngine::Layout::NHWC);
    InferenceEngine::Blob::Ptr imgBlob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, image.data);
    infer_request.SetBlob("input", imgBlob);

    /* Execute inference */
    infer_request.Infer();

    /* Read outputs */
    InferenceEngine::Blob::Ptr color = infer_request.GetBlob("color");
    InferenceEngine::Blob::Ptr type = infer_request.GetBlob("type");
    auto result_color = color->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    auto result_type = type->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    /* Dump outputs */
    std::cout << "Input image path: " << image_path << std::endl;
    std::cout << "    COLOR:" << std::endl;
    for (int i = 0, size = COLOR_LABEL.size(); i < size; i++) {
        std::cout << "        " << COLOR_LABEL[i] << " -> " << result_color[i] << std::endl;
    }
    std::cout << "    TYPE:" << std::endl;
    for (int i = 0, size = TYPE_LABEL.size(); i < size; i++) {
        std::cout << "        " << TYPE_LABEL[i] << " -> " << result_type[i] << std::endl;
    }

    return 0;
}
