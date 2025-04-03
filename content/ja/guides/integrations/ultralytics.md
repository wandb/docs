---
title: Ultralytics
menu:
  default:
    identifier: ja-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、姿勢推定などのタスクのための、最先端のコンピュータビジョン モデルの本拠地です。リアルタイムオブジェクト検出モデルの YOLO シリーズの最新版である [YOLOv8](https://docs.ultralytics.com/models/yolov8/) をホストするだけでなく、[SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)、[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) などの他の強力なコンピュータビジョン モデルもホストしています。これらのモデルの実装を提供するだけでなく、Ultralytics は、使いやすい API を使用してこれらのモデルをトレーニング、ファインチューン、および適用するための、すぐに使える ワークフローも提供します。

## 始めましょう

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

    開発チームは、`ultralyticsv8.0.238` 以下のバージョンとの インテグレーションをテストしました。インテグレーションに関する問題点を報告するには、`yolov8` タグを付けて [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## 実験管理の追跡と検証結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、トレーニング、ファインチューン、および検証に [Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使用し、[W&B](https://wandb.ai/site) を使用して実験管理の追跡、モデルのチェックポイント、およびモデルのパフォーマンスの可視化を実行する典型的なワークフローを示します。

この レポートで インテグレーションについて確認することもできます。[W&B で Ultralytics を強化する](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B インテグレーションを使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

選択した `YOLO` モデルを初期化し、モデルで推論を実行する前に、そのモデルで `add_wandb_callback` 関数を呼び出します。これにより、トレーニング、ファインチューン、検証、または推論を実行すると、実験 ログと画像が自動的に保存され、W&B 上の [コンピュータビジョン タスクのインタラクティブなオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) を使用して、グラウンドトゥルースとそれぞれの予測結果が重ねられ、追加の洞察が [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) にまとめられます。

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

Ultralytics のトレーニングまたはファインチューン ワークフローのために W&B を使用して追跡された実験は、次のようになります。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

[W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) を使用して、エポックごとの検証結果を可視化する方法を次に示します。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、推論に [Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使用し、[W&B](https://wandb.ai/site) を使用して結果を可視化する典型的なワークフローを示します。

Google Colab でコードを試すことができます:[Open in Colab](http://wandb.me/ultralytics-inference)。

この レポートで インテグレーションについて確認することもできます。[W&B で Ultralytics を強化する](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B インテグレーションを使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートする必要があります。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

インテグレーションをテストするために、いくつかの画像をダウンロードします。静止画像、ビデオ、またはカメラ ソースを使用できます。推論ソースの詳細については、[Ultralytics のドキュメント](https://docs.ultralytics.com/modes/predict/) を確認してください。

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

次に、目的の `YOLO` モデルを初期化し、モデルで推論を実行する前に、そのモデルで `add_wandb_callback` 関数を呼び出します。これにより、推論を実行すると、[コンピュータビジョン タスクのインタラクティブなオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) でオーバーレイされた画像が自動的にログに記録され、追加の洞察が [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) にまとめられます。

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

トレーニングまたはファインチューン ワークフローの場合、`wandb.init()` を使用して run を明示的に初期化する必要はありません。ただし、コードに予測のみが含まれる場合は、run を明示的に作成する必要があります。

インタラクティブな bbox オーバーレイは次のようになります。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B 画像オーバーレイの詳細については、[こちら]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ja" >}}) を参照してください。

## その他のリソース

* [Weights & Biases で Ultralytics を強化する](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [YOLOv8 を使用したオブジェクト検出: エンドツーエンドのワークフロー](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)
