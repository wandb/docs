---
title: Ultralytics
menu:
  default:
    identifier: ja-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、ポーズ推定など、さまざまなタスク向けの最先端コンピュータビジョンモデルの開発拠点です。[YOLOv8](https://docs.ultralytics.com/models/yolov8/)（リアルタイムオブジェクト検出モデルYOLOシリーズの最新バージョン）だけでなく、[SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)、[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) などの強力なモデルも提供しています。これらのモデルの実装だけでなく、直感的な API を使ったトレーニング、ファインチューニング、適用まで、エンドツーエンドで手軽に扱えるワークフローも提供されています。

## はじめよう

1. `ultralytics` と `wandb` をインストールします。

    {{< tabpane text=true >}}
    {{% tab header="コマンドライン" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # または
    # conda install ultralytics
    ```

    {{% /tab %}}
    {{% tab header="ノートブック" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    開発チームでは `ultralytics v8.0.238` 以下のバージョンでインテグレーションをテストしています。インテグレーションに関する不具合を報告したい場合は、タグに `yolov8` をつけて [GitHub Issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## 実験管理と検証結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを用いたトレーニング、ファインチューニング、検証の典型的なワークフロー、そして [W&B](https://wandb.ai/site) を活用した実験管理、モデルのチェックポイント保存、パフォーマンスの可視化について解説します。

インテグレーションの詳細な活用例はこのレポートもご覧ください: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B を使うには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

使いたい `YOLO` モデルを初期化し、推論の前に `add_wandb_callback` を呼び出します。これにより、トレーニング、ファインチューニング、検証、あるいは推論を実行した際、実験ログや画像（グラウンドトゥルースと予測結果の両方が重ね描きされたもの）が自動的に保存されます。また、[コンピュータビジョンタスク向けのインタラクティブオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) 付きで [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) に可視化情報が記録されます。

```python
with wandb.init(project="ultralytics", job_type="train") as run:

    # YOLOモデルを初期化
    model = YOLO("yolov8n.pt")

    # Ultralytics用のW&Bコールバックを追加
    add_wandb_callback(model, enable_model_checkpointing=True)

    # モデルのトレーニング・ファインチューニング
    # 各エポック後、検証バッチでの予測がW&Bテーブルにログされます
    # 可視化性に優れたインタラクティブなオーバーレイも作成されます
    model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)
```

W&B を使って Ultralytics でトラッキングした実験管理（トレーニング・ファインチューニング）の画面例はこちらです：

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

各エポックごとの検証結果が [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) でどのように可視化されるかもご覧ください：

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルによる推論ワークフロー、および [W&B](https://wandb.ai/site) を活用した結果の可視化例を紹介します。

Google Colab で試せるノートブックも用意しています：[Colab で開く](https://wandb.me/ultralytics-inference)

また、インテグレーション活用例の詳細はこのレポートでもご覧いただけます: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B を使うには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

連携動作を確認するために画像をいくつかダウンロードしましょう。静止画、動画、カメラ映像など様々なデータに対応できます。推論ソースの詳細は [Ultralytics のドキュメント](https://docs.ultralytics.com/modes/predict/) をご確認ください。

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

`wandb.init()` を使って W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化します。続いて、目的の `YOLO` モデルを初期化し、推論の前に `add_wandb_callback` を呼び出します。これにより推論時には、画像に[コンピュータビジョンタスク向けインタラクティブオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}})が自動付与され、[W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) へ追加情報とともに記録されます。

```python
# W&B Run を初期化
with wandb.init(project="ultralytics", job_type="inference") as run:
    # YOLOモデルを初期化
    model = YOLO("yolov8n.pt")

    # Ultralytics用のW&Bコールバックを追加
    add_wandb_callback(model, enable_model_checkpointing=True)

    # 推論を実行、W&B Tableへ自動記録
    # バウンディングボックスやセグメンテーションマスクのインタラクティブオーバーレイ付き
    model(
        [
            "./assets/img1.jpeg",
            "./assets/img3.png",
            "./assets/img4.jpeg",
            "./assets/img5.jpeg",
        ]
    )
```

トレーニングやファインチューニングの場合は `wandb.init()` を明示的に呼び出す必要はありませんが、推論だけを行う場合には run の明示的な作成が必要です。

インタラクティブなバウンディングボックスオーバーレイの例です：

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

詳細は [W&B 画像オーバーレイガイド]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ja" >}}) をご確認ください。

## その他のリソース

* [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)