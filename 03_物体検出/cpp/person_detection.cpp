#include <iostream>
#include <string>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <inference_engine.hpp>


int main(int argc, char** argv) {

    /* Check arguments */
    if (argc != 4) {
        std::cerr << "Usage: ./person_detection <model.xml> <model.bin> <image>" << std::endl;
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
    std::string input_blob_name = network.getInputsInfo().begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_info->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
    input_info->setLayout(InferenceEngine::Layout::NHWC);
    input_info->setPrecision(InferenceEngine::Precision::U8);
    std::cout << "input_blob_name: " << input_blob_name << std::endl;
    
    /* Configure output blobs */
    std::string output_blob_name = network.getOutputsInfo().begin()->first;
    InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
    const auto outputDims = output_info->getTensorDesc().getDims();
    const int maxProposalCount = outputDims[2];
    const int objectSize = outputDims[3];
    output_info->setPrecision(InferenceEngine::Precision::FP32);
    std::cout << "output_blob_name: " << output_blob_name << std::endl;

    /* Loading model to device */
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");

    /* Create infer request */
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
    infer_request.SetBlob(input_blob_name, imgBlob);

    /* Execute inference */
    infer_request.Infer();

    /* Read outputs */
    InferenceEngine::Blob::Ptr outputs = infer_request.GetBlob(output_blob_name);
    const auto detection = outputs->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    /* Each detection has image_id that denotes processed image */
    std::cout << "Input image path: " << image_path << std::endl;
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        const auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
        if (image_id < 0) {
            break;
        }

        const auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
        const auto confidence = detection[curProposal * objectSize + 2];
        const auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * image.size().width);
        const auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * image.size().height);
        const auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * image.size().width);
        const auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * image.size().height);

        std::cout << "    label=" << label << "  confidence=" << confidence 
                  << "  location=(" << xmin << "," << ymin << "," << xmax << "," << ymax << ")" << std::endl;
        
        if (confidence > 0.5) {
            cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 0, 255), 1);
        }
    }

    const bool is_success = cv::imwrite("result.jpg", image);
    if (is_success) {
        std::cout << "dump: result.jpg" << std::endl;
    }
    
    return 0;
}
