---
title: YOLOv5
menu:
  default:
    identifier: ja-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics の YOLOv5](https://ultralytics.com/yolo)（"You Only Look Once"）モデルファミリーは、畳み込みニューラルネットワークを使ってリアルタイムでオブジェクト検出を実現します。面倒な作業なしで使えるのも魅力です。

[W&B](https://wandb.com) は YOLOv5 に直接インテグレーションされており、実験メトリクスのトラッキング、モデルやデータセットのバージョン管理、リッチなモデル予測の可視化など、さまざまな機能をサポートします。**YOLO 実験を始める前に `pip install` を一回実行するだけで、すぐに使い始められます。**

{{% alert %}}
すべての W&B ログ機能は、[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) などのデータ並列マルチ GPU トレーニングでも利用できます。
{{% /alert %}}

## コア実験のトラッキング
`wandb` をインストールするだけで、W&B の[ログ機能]({{< relref path="/guides/models/track/log/" lang="ja" >}})が有効になります。これにより、システムメトリクス・モデルメトリクス・メディアがインタラクティブな [Dashboards]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) に記録されます。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さなネットワークを小さなデータセットでトレーニング
```

実行後、wandb が標準出力に表示するリンクをクリックしてください。

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="このほかにもたくさんのチャートが表示されます。" >}}

## インテグレーションのカスタマイズ

YOLO のコマンドライン引数に少し手を加えるだけで、さらに多くの W&B 機能が利用可能です。

* `--save_period` に数値を指定すると、W&B は毎回 `save_period` エポックごとに [model version]({{< relref path="/guides/core/registry/" lang="ja" >}}) を保存します。モデルの重みも含まれ、検証セットで最も良いスコアのモデルにタグが付きます。
* `--upload_dataset` フラグを追加すると、データセットのアップロードも行われ、データのバージョン管理ができます。
* `--bbox_interval` に数値を指定すると、[data visualization]({{< relref path="../" lang="ja" >}}) 機能が有効になります。各 `bbox_interval` エポック終了時に、モデルの検証セットに対する出力が W&B へアップロードされます。

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
W&B アカウントには、データセットやモデル用に 100GB の無料ストレージが付いています。
{{% /alert %}}

実際の画面イメージはこちら。

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="Model versioning" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="Data visualization" >}}

{{% alert %}}
データとモデルのバージョン管理によって、一時停止した実験やクラッシュした実験も、どのデバイスからでもセットアップ不要ですぐに再開できます。詳しくは [Colab の例](https://wandb.me/yolo-colab) をご覧ください。
{{% /alert %}}