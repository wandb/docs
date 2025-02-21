---
title: Ultralytics
menu:
  default:
    identifier: ja-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、ポーズ推定などのタスクに対応する、最先端のコンピュータビジョンモデルの本拠地です。リアルタイムオブジェクト検出モデルである YOLO シリーズの最新版 [YOLOv8](https://docs.ultralytics.com/models/yolov8/) をホストするだけでなく、[SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) などの強力なコンピュータビジョンモデルもホストしています。これらのモデルの実装を提供するだけでなく、Ultralytics は、使いやすい API を使用してこれらのモデルのトレーニング、ファインチューニング、および適用を行うための、すぐに使える ワークフロー も提供しています。

## はじめに

1. `ultralytics` と `wandb` をインストールします。

    {{< tabpane text=true >}}
    {{% tab header="コマンドライン" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # or
    # conda install ultralytics
    ```

    {{% /tab %}}
    {{% tab header="Notebook" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    開発チームは、`ultralyticsv8.0.238` 以下との インテグレーション をテストしました。 インテグレーション に関する問題をご報告いただくには、`yolov8` というタグを付けて [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## Experiments を追跡し、検証結果を可視化する

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルをトレーニング、ファインチューニング、および検証に使用し、[W&B](https://wandb.ai/site) を使用して 実験管理 、モデル チェックポイント 、およびモデルのパフォーマンスの 可視化 を行う典型的な ワークフロー を示します。

この report の インテグレーション についても確認できます: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B インテグレーション を使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

選択した `YOLO` モデルを初期化し、モデルで推論を実行する前に、そのモデルで `add_wandb_callback` 関数を呼び出します。これにより、トレーニング、ファインチューニング、検証、または推論を実行すると、 実験 ログと画像が自動的に保存され、[コンピュータビジョンタスク用のインタラクティブオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) を使用して、グラウンドトゥルースとそれぞれの予測結果が W&B 上にオーバーレイされます。さらに、[`wandb.Table`]({{< relref path="/guides/core/tables/" lang="ja" >}}) に追加のインサイトが表示されます。

```python
# Initialize YOLO Model
model = YOLO("yolov8n.pt")

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train/fine-tune your model
# At the end of each epoch, predictions on validation batches are logged
# to a W&B table with insightful and interactive overlays for
# computer vision tasks
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# Finish the W&B run
wandb.finish()
```

Ultralytics のトレーニング または ファインチューニング ワークフロー で W&B を使用して追跡された Experiments は次のようになります。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

エポックごとの検証結果が [W&B Table]({{< relref path="/guides/core/tables/" lang="ja" >}}) を使用して 可視化 される方法は次のとおりです。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果を 可視化 する

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、推論に [Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使用し、[W&B](https://wandb.ai/site) を使用して結果を 可視化 する典型的な ワークフロー を示します。

Google Colab でコードを試すことができます: [Open in Colab](http://wandb.me/ultralytics-inference).

この report の インテグレーション についても確認できます: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B インテグレーション を使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートする必要があります。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

インテグレーション をテストするために、いくつかの画像をダウンロードします。静止画像、ビデオ、またはカメラソースを使用できます。推論ソースの詳細については、[Ultralytics ドキュメント](https://docs.ultralytics.com/modes/predict/) を参照してください。

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

次に、`wandb.init` を使用して W&B [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化します。

```python
# Initialize W&B run
wandb.init(project="ultralytics", job_type="inference")
```

次に、目的の `YOLO` モデルを初期化し、モデルで推論を実行する前に、そのモデルで `add_wandb_callback` 関数を呼び出します。これにより、推論を実行すると、[コンピュータビジョンタスク用のインタラクティブオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) でオーバーレイされた画像が自動的に ログ に記録され、[`wandb.Table`]({{< relref path="/guides/core/tables/" lang="ja" >}}) に追加のインサイトが表示されます。

```python
# Initialize YOLO Model
model = YOLO("yolov8n.pt")

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Perform prediction which automatically logs to a W&B Table
# with interactive overlays for bounding boxes, segmentation masks
model(
    [
        "./assets/img1.jpeg",
        "./assets/img3.png",
        "./assets/img4.jpeg",
        "./assets/img5.jpeg",
    ]
)

# Finish the W&B run
wandb.finish()
```

トレーニング または ファインチューニング ワークフロー の場合、`wandb.init()` を使用して run を明示的に初期化する必要はありません。ただし、コードに予測のみが含まれる場合は、run を明示的に作成する必要があります。

インタラクティブな bbox オーバーレイは次のようになります。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B 画像オーバーレイの詳細については、[こちら]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ja" >}}) をご覧ください。

## その他のリソース

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)
