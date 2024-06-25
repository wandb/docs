---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# YOLOv5

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5)（"You Only Look Once"）モデルファミリーは、畳み込みニューラルネットワークを使用してリアルタイムのオブジェクト検出を可能にし、煩わしい作業をすべて省きます。

[Weights & Biases](http://wandb.com) は YOLOv5 に直接統合されており、実験メトリクスの追跡、モデルとデータセットのバージョン管理、豊富なモデル予測の可視化などを提供します。**YOLOの実験を行う前に `pip install` を実行するだけで簡単に始められます！**

:::info
YOLOv5 インテグレーションのモデルとデータロギング機能についての簡単な概要は、[この Colab](https://wandb.me/yolo-colab) と以下のビデオチュートリアルをご覧ください。
:::

:::info
すべての W&B ロギング機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp\_tutorial.html) などのデータ並列マルチGPUトレーニングと互換性があります。
:::

## Core Experiment Tracking

`wandb` をインストールするだけで、システムメトリクス、モデルメトリクス、およびインタラクティブな[ダッシュボード](../track/app.md)にログを記録するための組み込み W&B [ロギング機能](../track/log/intro.md)が有効になります。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなネットワークを小さなデータセットでトレーニング
```

wandb によって標準出力に表示されるリンクに従うだけです。

![これらのチャートとさらに多くのもの！](/images/integrations/yolov5_experiment_tracking.png)

## Model Versioning and Data Visualization

しかし、それだけではありません！YOLO にいくつかの簡単なコマンドライン引数を渡すことで、さらに多くの W&B 機能を利用できます。

* `--save_period` に数値を渡すと、[モデルバージョン管理](../model_registry/intro.md)が有効になります。`save_period` エポックごとにモデルの重みが W&B に保存されます。検証セットで最も性能の良いモデルが自動的にタグ付けされます。
* `--upload_dataset` フラグを有効にすると、データセットもアップロードされ、データバージョン管理が可能になります。
* `--bbox_interval` に数値を渡すと、[データ可視化](../intro.md)が有効になります。`bbox_interval` エポックごとに、検証セットのモデル出力が W&B にアップロードされます。

<Tabs
  defaultValue="modelversioning"
  values={[
    {label: 'Model Versioning Only', value: 'modelversioning'},
    {label: 'Model Versioning and Data Visualization', value: 'bothversioning'},
  ]}>
  <TabItem value="modelversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1
```

  </TabItem>
  <TabItem value="bothversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

  </TabItem>
</Tabs>

:::info
すべての W&B アカウントには、データセットとモデルのための 100 GB の無料ストレージが付いています。
:::

このようになります。

![Model Versioning: 最新バージョンと最良バージョンのモデルが特定されます。](/images/integrations/yolov5_model_versioning.png)

![Data Visualization: 入力画像とモデルの出力、および各例に対するメトリクスを比較します。](/images/integrations/yolov5_data_visualization.png)

:::info
データとモデルのバージョン管理を使用して、中断したりクラッシュしたりした実験を任意のデバイスから再開できます。詳細は[この Colab](https://wandb.me/yolo-colab) をご覧ください。
:::
