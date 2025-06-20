---
title: YOLOv5
menu:
  default:
    identifier: ja-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics' YOLOv5](https://ultralytics.com/yolo) ("You Only Look Once") モデルファミリーは、畳み込みニューラルネットワークを使用したリアルタイムのオブジェクト検出を、苦痛なく実現します。

[Weights & Biases](http://wandb.com) は YOLOv5 に直接インテグレーションされており、実験のメトリクス追跡、モデルとデータセットのバージョン管理、リッチなモデル予測の可視化などを提供します。**YOLO の実験を実行する前に `pip install` 一行を実行するだけで始められます。**

{{% alert %}}
すべての W&B ログ機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などのデータ並列マルチGPUトレーニングと互換性があります。
{{% /alert %}}

## コア実験の追跡

`wandb` をインストールするだけで、システムメトリクス、モデルメトリクス、インタラクティブな[ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})にログされるメディアといった、ビルトインの W&B [ログ機能]({{< relref path="/guides/models/track/log/" lang="ja" >}})が有効になります。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなデータセットで小さなネットワークをトレーニングします
```

wandb によって標準出力に表示されるリンクをただフォローするだけです。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="これらのチャートおよびそれ以上。" >}}

## インテグレーションのカスタマイズ

YOLO にいくつかの簡単なコマンドライン引数を渡すことで、さらに多くの W&B 機能を活用できます。

* `--save_period` に数値を渡すと、W&B は各 `save_period` エポックの終わりに[モデルバージョン]({{< relref path="/guides/core/registry/" lang="ja" >}})を保存します。モデルバージョンにはモデルの重みが含まれ、検証セットで最もパフォーマンスの良いモデルにタグ付けされます。
* `--upload_dataset` フラグをオンにすると、データセットがデータバージョン管理のためにアップロードされます。
* `--bbox_interval` に数値を渡すと、[データ可視化]({{< relref path="../" lang="ja" >}})が有効になります。各 `bbox_interval` エポックの終わりに、モデルの出力が検証セットに対して W&B にアップロードされます。

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

これがどのように見えるかを示します。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="モデルバージョン管理: 最新かつベストなモデルバージョンが識別されます。" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="データ可視化: 入力画像とモデルの出力および例ごとのメトリクスを比較します。" >}}

{{% alert %}}
データとモデルのバージョン管理により、セットアップ不要で任意のデバイスから一時停止またはクラッシュした実験を再開できます。[詳細は Colab を確認してください](https://wandb.me/yolo-colab)。
{{% /alert %}}