---
title: YOLOv5
menu:
  default:
    identifier: ja-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5) ("You Only Look Once") モデルファミリーは、畳み込みニューラルネットワークを用いたリアルタイムオブジェクト検出を簡単に実現します。

[Weights & Biases](http://wandb.com) は YOLOv5 に直接インテグレーションされており、実験のメトリクストラッキング、モデルとデータセットのバージョン管理、リッチなモデル予測の可視化などを提供します。**YOLO 実験を実行する前に `pip install` を1回実行するだけで始められます。**

{{% alert %}}
すべての W&B ログ機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などのデータ・パラレルなマルチ GPU トレーニングと互換性があります。
{{% /alert %}}

## コア実験をトラックする
`wandb` をインストールするだけで、W&B の組み込み [ログ機能]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を有効にできます: システムメトリクス、モデルメトリクス、インタラクティブな [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) にログされるメディアなどです。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなデータセットで小さなネットワークをトレーニング
```

wandb によって標準出力に表示されるリンクに従うだけです。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="これらのチャートともっと。" >}}

## インテグレーションをカスタマイズする

YOLO にいくつかの簡単なコマンドライン引数を渡すことで、さらに多くの W&B 機能を活用できます。

* `--save_period` に数値を渡すことで [モデルのバージョン管理]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) をオンにできます。`save_period` エポックごとに、モデルの重みが W&B に保存されます。検証セットで最も良い性能を発揮したモデルには自動的にタグが付きます。
* `--upload_dataset` フラグをオンにすると、データセットをアップロードしてデータのバージョン管理を行うことができます。
* `--bbox_interval` に数値を渡すことで [データ可視化]({{< relref path="../" lang="ja" >}}) をオンにできます。`bbox_interval` エポックごとに、モデルの検証セット上での出力が W&B にアップロードされます。

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
すべての W&B アカウントには、データセットとモデル用に 100 GB の無料ストレージが付属しています。
{{% /alert %}}

こちらがその様子です。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="モデルのバージョン管理: 最新かつ最高のバージョンのモデルが特定されています。" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="データ可視化: 入力画像とモデルの出力、例ごとのメトリクスを比較します。" >}}

{{% alert %}}
データとモデルのバージョン管理を利用すると、どのデバイスからでも一時停止またはクラッシュした実験を再開できます。詳細は [the Colab](https://wandb.me/yolo-colab) をご覧ください。
{{% /alert %}}