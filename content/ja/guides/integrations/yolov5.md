---
title: YOLOv5
menu:
  default:
    identifier: ja-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5) (「You Only Look Once」) モデルファミリーは、苦痛を伴うことなく、畳み込みニューラルネットワークによるリアルタイムの オブジェクト検出を可能にします。

[Weights & Biases](http://wandb.com) は YOLOv5 に直接統合されており、実験の メトリクス 追跡、モデルと データセット の バージョン管理 、豊富なモデル 予測 の 可視化 などを提供します。**YOLO の 実験 を実行する前に、`pip install` を 1 回実行するだけで簡単に利用できます。**

{{% alert %}}
W&B のすべての ログ 機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などの データ並列マルチ GPU トレーニングと互換性があります。
{{% /alert %}}

## コアな 実験 を追跡する
`wandb` をインストールするだけで、組み込みの W&B [ログ 機能]({{< relref path="/guides/models/track/log/" lang="ja" >}}) が有効になります。システム メトリクス 、モデル メトリクス 、および インタラクティブな [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) に ログ されるメディアです。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなデータセットで小さなネットワークをトレーニングします。
```

wandb によって標準出力に出力されたリンクをたどってください。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="All these charts and more." >}}

## インテグレーション をカスタマイズする

いくつかの簡単な コマンドライン 引数 を YOLO に渡すことで、さらに多くの W&B 機能を活用できます。

* `--save_period` に数値を渡すと、W&B は `save_period` エポック の終了ごとに [モデル バージョン]({{< relref path="/guides/core/registry/" lang="ja" >}}) を保存します。モデル バージョン には、モデルの 重み が含まれており、 検証セット で最高のパフォーマンスを発揮するモデルにタグを付けます。
* `--upload_dataset` フラグをオンにすると、 データ バージョン管理 のために データセット もアップロードされます。
* `--bbox_interval` に数値を渡すと、[データ可視化]({{< relref path="../" lang="ja" >}}) が有効になります。`bbox_interval` エポック の終了ごとに、 検証セット 上のモデルの出力が W&B にアップロードされます。

{{< tabpane text=true >}}
{{% tab header="モデルの バージョン管理 のみ" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="モデルの バージョン管理 と データ可視化" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
すべての W&B アカウントには、 データセット と モデル 用に 100 GB の無料ストレージが付属しています。
{{% /alert %}}

このようになります。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="Model Versioning: the latest and the best versions of the model are identified." >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="Data Visualization: compare the input image to the model's outputs and example-wise metrics." >}}

{{% alert %}}
データ と モデル の バージョン管理 により、セットアップなしで、一時停止またはクラッシュした 実験 を任意のデバイスから再開できます。詳細については、[Colab ](https://wandb.me/yolo-colab) をご覧ください。
{{% /alert %}}
