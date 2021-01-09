# OpenVINO

[OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)の勉強のアウトプットを残します。

## OpenVINOとは

OpenVINO ToolkitはIntel社が公開しているディープラーニングフレームワークで、最適化されたディープラーニングモデルのための推論エンジンを利用できます。また、OpenVINOはモデル学習機能を持ちませんが、TensorFlow、PyTorchなどで学習したモデルをOpenVINOで実行可能な形式に変換できるモデルオプティマイザ機能が実装されています。

一般的にディープラーニングの推論はGPU上で演算しますが、OpenVINOはIntel社製CPUで高速に実行できるように最適化されています。したがって、用意するハードウェアにGPUは不要です。

OpenVINOを実行可能なプログラミング言語はPythonとC++です。ここでは両方のコードを公開します。OpenVINOは、Linux、Windows、MacOSで動作し、ここではLinux上で開発します。

## 前提

- CentOS 7.9.2009
- OpenVINO 2020.4
- Python 3.7.7
- GCC 4.8.5
- PyTorch 1.7.0

## 目次

1. [インストール](https://github.com/JuvenileTalk9/OpenVINO/01_インストール/インストール.md)
2. [画像認識](https://github.com/JuvenileTalk9/OpenVINO/02_画像認識/画像認識.md)

（作成中）
