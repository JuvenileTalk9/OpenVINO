import os
import sys
import logging
from argparse import ArgumentParser

import cv2
import numpy as np
from openvino.inference_engine import IECore


MODEL_XML = '/opt/intel/openvino/deployment_tools/tools/model_downloader/intel/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.xml'
MODEL_BIN = '/opt/intel/openvino/deployment_tools/tools/model_downloader/intel/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.bin'

COLOR_LABEL = ('white', 'gray', 'yellow', 'red', 'green', 'blue', 'black')
TYPE_LABEL = ('car', 'van', 'truck', 'bus')


def build_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG, stream=sys.stdout)
    return logger


def build_argparser():
    parser = ArgumentParser(description='Recognize vehicle attributes.')
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
    n, c, h, w = net.input_info['input'].input_data.shape
    # n, c, h, w = net.inputs['input'].shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.inputs[i])
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # HWC から CHW へ変換
        images[i, :, :, :] = image

    # データを入力して推論を実行する
    exec_net = ie.load_network(network=net, device_name='CPU')
    res = exec_net.infer(inputs={'input': images})
    logger.info('Inference completed.')

    # 出力層から結果を得る
    res_color, res_type = res['color'], res['type']
    logger.debug('res_color: {}'.format(res_color))
    logger.debug('res_type: {}'.format(res_type))

    # 判定結果を出力する
    for i in range(len(res_color)):
        logger.info('--------------------------------------')
        logger.info('FILE: {}'.format(args.inputs[i]))
        logger.info('COLOR: {}'.format(COLOR_LABEL[np.argmax(res_color[i])]))
        logger.info('TYPE: {}'.format(TYPE_LABEL[np.argmax(res_type[i])]))
    logger.info('--------------------------------------')
