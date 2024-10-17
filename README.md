# OpenVINO

OpenVINO Toolkit（Intelが提供するディープラーニングモデルを実行・最適化するツール）のサンプルをまとめたものです。

|サンプル|内容|入力|出力|
|:--|:--|:--|:--|
|[person-detection-0303](https://github.com/JuvenileTalk9/OpenVINO/tree/main/sample/person-detection-0303)|人検知|`B, C, H, W`|boxes: `N, 5`、labels: `N`|
|[vehicle-detection-0202](https://github.com/JuvenileTalk9/OpenVINO/tree/main/sample/vehicle-detection-0202)|車検知|`B, C, H, W`|`1, 1, N, 7`|
|[human-pose-estimation-0007](https://github.com/JuvenileTalk9/OpenVINO/tree/main/sample/human-pose-estimation-0007)|骨格抽出|`B, C, H, W`|`1, 17, 224, 224`|
