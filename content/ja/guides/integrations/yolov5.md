---
title: YOLOv5
menu:
  default:
    identifier: ja-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5)（"You Only Look Once"）モデルファミリーは、苦痛を伴わずに畳み込みニューラルネットワークによるリアルタイムの object detection を実現します。

[Weights & Biases](http://wandb.com) は YOLOv5 に直接統合されており、実験のメトリクスのトラッキング、モデル と dataset の バージョン管理、リッチなモデル 予測 の 可視化 などを提供します。**YOLO の実験 を実行する前に `pip install` を 1 回実行するだけで、簡単に利用できます。**

{{% alert %}}
W&B のすべてのログ機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などのデータ並列マルチ GPU トレーニング と互換性があります。
{{% /alert %}}

## コアとなる実験 を追跡する
`wandb` をインストールするだけで、組み込みの W&B [ログ機能]({{< relref path="/guides/models/track/log/" lang="ja" >}}) が有効になります。システムメトリクス、モデルメトリクス、およびインタラクティブな [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) に記録されたメディア。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなデータセットで小さなネットワークをトレーニングします
```

wandb によって標準出力に出力されたリンクをたどってください。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="All these charts and more." >}}

## インテグレーションをカスタマイズする

いくつかの簡単な コマンドライン 引数 を YOLO に渡すことで、さらに多くの W&B 機能を利用できます。

* `--save_period` に数値を渡すと、[モデル の バージョン管理]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) が有効になります。`save_period` エポック が終了するたびに、モデル の weights が W&B に保存されます。検証セット で最高のパフォーマンスを発揮するモデル には、自動的にタグが付けられます。
* `--upload_dataset` フラグをオンにすると、データ バージョン管理 用に dataset もアップロードされます。
* `--bbox_interval` に数値を渡すと、[data visualization]({{< relref path="../" lang="ja" >}}) が有効になります。`bbox_interval` エポック が終了するたびに、検証セット でのモデル の出力が W&B にアップロードされます。

{{< tabpane text=true >}}
{{% tab header="モデル の バージョン管理 のみ" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="モデル の バージョン管理 と Data Visualization" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
すべての W&B アカウントには、データセット と モデル 用に 100 GB の無料ストレージが付属しています。
{{% /alert %}}

それがどのように見えるかは次のとおりです。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="モデル の バージョン管理：モデル の最新バージョン と 最良バージョン が識別されます。" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="Data Visualization：入力画像 と モデル の出力、およびサンプルごとのメトリクスを比較します。" >}}

{{% alert %}}
データ と モデル の バージョン管理 により、セットアップなしで、一時停止またはクラッシュした実験 を任意のデバイスから再開できます。詳細については、[Colab ](https://wandb.me/yolo-colab) をご覧ください。
{{% /alert %}}
