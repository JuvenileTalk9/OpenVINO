import os
import sys
import logging
from argparse import ArgumentParser

import cv2
import numpy as np
from openvino.inference_engine import IECore


MODEL_XML = '/opt/intel/openvino/deployment_tools/tools/model_downloader/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml'
MODEL_BIN = '/opt/intel/openvino/deployment_tools/tools/model_downloader/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin'

INPUT_LAYER_NAME, OUTPUT_LAYER_NAME = 'data', 'prob_emotion'

EMOTION_LABEL = ('neutral', 'happy', 'sad', 'surprise', 'anger')


def build_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG, stream=sys.stdout)
    return logger


def build_argparser():
    parser = ArgumentParser(description='Recognize emotions.')
    parser.add_argument('-i', '--inputs', required=True, type=str, nargs='+', help='[Required] path to image files')
    return parser


if __name__ == '__main__':

    # ロガーを生成する
    logger = build_logger()

    # 引数を読み込む
    args = build_argparser().parse_args()

    # モデルをロードする
    ie = IECore()
    net = ie.read_network(model=MODEL_XML, weights=MODEL_BIN)
    logger.info('Successfully loaded model.')

    # バッチサイズを指定する
    net.batch_size = len(args.inputs)
    logger.info('batch size: {}'.format(net.batch_size))

    # 入力データを生成する
    try:
        n, c, h, w = net.input_info[INPUT_LAYER_NAME].input_data.shape
        images = np.ndarray(shape=(n, c, h, w))
        for i in range(n):
            image = cv2.imread(args.inputs[i])
            image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))  # HWC から CHW へ変換
            images[i, :, :, :] = image
    except KeyError as e:
        logger.error('Undefined layer name: {}'.format(e))
        logger.error('Defined layer names: {}'.format([key for key in net.layers.keys()]))
        exit(1)

    # データを入力して推論を実行する
    exec_net = ie.load_network(network=net, device_name='CPU')
    res = exec_net.infer(inputs={INPUT_LAYER_NAME: images})
    logger.info('Inference completed.')

    # 出力層から結果を得る
    try:
        res_emotion = res[OUTPUT_LAYER_NAME]
    except KeyError as e:
        logger.error('Undefined layer name: {}'.format(e))
        logger.error('Defined layer names: {}'.format([key for key in net.layers.keys()]))
        exit(1)

    # 判定結果を出力する
    logger.info('---------- INFERENCE RESULT ----------')
    for i in range(len(res_emotion)):
        logger.info('FILE: {} \t EMOTION: {}'.format(args.inputs[i], EMOTION_LABEL[np.argmax(res_emotion[i])]))
    logger.info('--------------------------------------')
    logger.info('Process completed.')