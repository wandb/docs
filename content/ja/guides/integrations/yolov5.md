---
title: YOLOv5
menu:
  default:
    identifier: yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics の YOLOv5](https://ultralytics.com/yolo)（"You Only Look Once"）モデルファミリーは、畳み込みニューラルネットワークを使ってリアルタイムのオブジェクト検出を手軽に実現します。

[W&B](https://wandb.com) は YOLOv5 に直接インテグレーションされており、実験メトリクスのトラッキング、モデルやデータセットのバージョン管理、リッチなモデル予測の可視化などが簡単に利用できます。**YOLO 実験を実行する前に `pip install` を一回実行するだけで始められます。**

{{% alert %}}
すべての W&B ロギング機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などのデータ並列型マルチ GPU トレーニングにも対応しています。
{{% /alert %}}

## コア実験をトラッキングする
`wandb` をインストールするだけで、W&B の [ロギング機能]({{< relref "/guides/models/track/log/" >}})（システムメトリクス、モデルメトリクス、インタラクティブな [ダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}}) へのメディア保存）が有効になります。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小規模ネットワークを小さなデータセットでトレーニング
```

wandb によって標準出力に表示されるリンクをクリックするだけです。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="これらのチャート以外にも、多くの情報を確認できます。" >}}

## インテグレーションをカスタマイズする

YOLO にいくつかの簡単なコマンドライン引数を追加するだけで、さらに多くの W&B 機能を活用できます。

* `--save_period` に数値を指定すると、W&B は毎回 `save_period` エポック終了時に [モデルバージョン]({{< relref "/guides/core/registry/" >}}) を保存します。モデルバージョンにはモデル重みが含まれ、検証セットで最も良い結果を出したモデルもタグ付けされます。
* `--upload_dataset` フラグを有効にすると、データセットもアップロードされ、データのバージョン管理が行われます。
* `--bbox_interval` に数値を指定すると、[データ可視化]({{< relref "../" >}}) が有効になります。すべての `bbox_interval` エポック終了時に、検証セットでのモデルの出力が W&B にアップロードされます。

{{< tabpane text=true >}}
{{% tab header="Model Versioning Only" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="Model Versioning and Data Visualization" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
W&B アカウントにはデータセットやモデルのための 100 GB の無料ストレージが付いてきます。
{{% /alert %}}

このようなイメージになります。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="モデルのバージョン管理" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="データ可視化" >}}

{{% alert %}}
データとモデルのバージョン管理によって、一時停止やクラッシュした実験もどのデバイスからでも再開できます。詳しくは [Colab ](https://wandb.me/yolo-colab) をご覧ください。
{{% /alert %}}