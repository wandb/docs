---
title: YOLOv5
menu:
  default:
    identifier: ja-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics の YOLOv5](https://ultralytics.com/yolo)（「You Only Look Once」）モデル ファミリーは、煩雑さなしに畳み込みニューラルネットワークでリアルタイムのオブジェクト検出を可能にします。

[W&B](https://wandb.com) は YOLOv5 に直接インテグレーションされており、実験のメトリクス追跡、モデルとデータセットのバージョン管理、リッチなモデル予測の可視化などを提供します。**YOLO の実験を実行する前に `pip install` を 1 回実行するだけで OK です。**

{{% alert %}}
すべての W&B のログ機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などのデータ並列マルチ GPU トレーニングと互換性があります。
{{% /alert %}}

## コアな実験をトラッキング
`wandb` をインストールするだけで、W&B の組み込みの[ログ機能]({{< relref path="/guides/models/track/log/" lang="ja" >}})が有効になります。システム メトリクス、モデル メトリクスに加え、インタラクティブな[ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})にメディアが記録されます。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなデータセットで小規模なネットワークをトレーニング
```

wandb が標準出力に表示するリンクに従うだけです。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="これらのチャートや他にもいろいろ。" >}}

## インテグレーションをカスタマイズ

YOLO にいくつかの簡単なコマンドライン引数を渡せば、さらに多くの W&B 機能を活用できます。

* `--save_period` に数値を渡すと、W&B は各 `save_period` エポックの最後に[モデル バージョン]({{< relref path="/guides/core/registry/" lang="ja" >}})を保存します。このモデル バージョンにはモデルの重みが含まれ、検証セットで最も性能が高いモデルにタグ付けします。
* フラグ `--upload_dataset` を有効にすると、データ バージョン管理のためにデータセットもアップロードされます。
* `--bbox_interval` に数値を渡すと[データ可視化]({{< relref path="../" lang="ja" >}})が有効になります。各 `bbox_interval` エポックの最後に、検証セットにおけるモデルの出力が W&B にアップロードされます。

{{< tabpane text=true >}}
{{% tab header="モデルのバージョン管理のみ" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="モデルのバージョン管理とデータ可視化" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
すべての W&B アカウントには、データセットとモデル向けに 100 GB の無料ストレージが付属します。
{{% /alert %}}

実際の見た目はこんな感じです。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="モデルのバージョン管理" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="データ可視化" >}}

{{% alert %}}
データとモデルのバージョン管理があれば、中断したりクラッシュした実験を、セットアップ不要でどのデバイスからでも再開できます。詳しくは [Colab ](https://wandb.me/yolo-colab) をご覧ください。
{{% /alert %}}