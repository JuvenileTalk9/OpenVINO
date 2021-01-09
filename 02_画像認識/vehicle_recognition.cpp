# 画像認識

ここではCNNなどの画像認識を事前学習モデルで実行する手順を示します。今回は車判定モデルを使います。

使用する事前学習モデルは以下の車判定モデルを使用します。

[https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/description/vehicle-attributes-recognition-barrier-0042.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/description/vehicle-attributes-recognition-barrier-0042.md)

モデルはOpenVINOのモデルダウンローダを使用します。

```sh
$ cd /opt/intel/openvino_2020.4.287/deployment_tools/open_model_zoo/tools/downloader
$ sudo python3 downloader.py --name vehicle-attributes-recognition-barrier-0042 --precisions FP32
```

## Python



## C++


[目次へ戻る](https://github.com/JuvenileTalk9/OpenVINO)
