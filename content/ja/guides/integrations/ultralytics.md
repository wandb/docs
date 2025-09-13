---
title: Ultralytics
menu:
  default:
    identifier: ja-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、姿勢推定といったタスク向けの 最先端の コンピュータビジョン モデルの拠点です。リアルタイム オブジェクト検出モデル YOLO シリーズの最新である [YOLOv8](https://docs.ultralytics.com/models/yolov8/) だけでなく、[SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)、[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) など強力なモデルも提供しています。さらに Ultralytics は、これらのモデルを使ったトレーニング、ファインチューニング、適用のための すぐに使える ワークフローと、使いやすい API も提供しています。

## Get started

1. `ultralytics` と `wandb` をインストールします。

    {{< tabpane text=true >}}
    {{% tab header="コマンドライン" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # あるいは
    # conda で ultralytics をインストール
    ```

    {{% /tab %}}
    {{% tab header="ノートブック" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    開発チームはこのインテグレーションを `ultralytics v8.0.238` 以前でテスト済みです。問題を報告する場合は、タグ `yolov8` を付けて [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## Track experiments and visualize validation results

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) のモデルを使ったトレーニング、ファインチューニング、検証の典型的なワークフローを紹介し、[W&B](https://wandb.ai/site) で 実験管理、モデルのチェックポイント作成、性能の可視化 を行う方法を示します。

このインテグレーションの詳細は次の W&B Report もご覧ください: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B のインテグレーションを使うには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

任意の `YOLO` モデルを初期化し、推論を実行する前に `add_wandb_callback` 関数を呼び出します。これにより、トレーニング、ファインチューニング、検証、推論を行うたびに、実験のログや、正解データと対応する予測結果を重ね合わせた画像が、自動的に W&B 上の [コンピュータビジョンタスク向けインタラクティブ オーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) とともに保存され、さらに [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) に追加のインサイトが記録されます。

```python
with wandb.init(project="ultralytics", job_type="train") as run:

    # YOLO モデルを初期化
    model = YOLO("yolov8n.pt")

    # Ultralytics 用の W&B コールバックを追加
    add_wandb_callback(model, enable_model_checkpointing=True)

    # モデルをトレーニング / ファインチューニング
    # 各エポックの最後に、検証バッチでの予測をログします
    # コンピュータビジョン向けの洞察的でインタラクティブなオーバーレイ付きの W&B Table に
    model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)
```

以下は、Ultralytics のトレーニング / ファインチューニング ワークフローで W&B により追跡された実験がどのように見えるかの例です:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO ファインチューニング Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

以下は、各エポックごとの検証結果が [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) でどのように可視化されるかの例です:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## Visualize prediction results

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) のモデルで推論を行い、結果を [W&B](https://wandb.ai/site) で可視化する典型的なワークフローを紹介します。

Google Colab でコードを試せます: [Open in Colab](https://wandb.me/ultralytics-inference)

このインテグレーションの詳細は次の W&B Report もご覧ください: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

W&B を Ultralytics と連携して使うには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

インテグレーションを試すための画像をいくつかダウンロードします。静止画、動画、カメラ入力を使用できます。推論ソースの詳細は [Ultralytics docs](https://docs.ultralytics.com/modes/predict/) を参照してください。

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

`wandb.init()` を使って W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化します。次に、使用したい `YOLO` モデルを初期化し、推論を実行する前に `add_wandb_callback` 関数を呼び出します。こうすることで、推論時に画像は自動的に [コンピュータビジョンタスク向けインタラクティブ オーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) を重ねた状態でログされ、[`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) に追加のインサイトとともに記録されます。

```python
# W&B の Run を初期化
with wandb.init(project="ultralytics", job_type="inference") as run:
    # YOLO モデルを初期化
    model = YOLO("yolov8n.pt")

    # Ultralytics 用の W&B コールバックを追加
    add_wandb_callback(model, enable_model_checkpointing=True)

    # 予測を実行すると、W&B Table に自動でログされます
    # バウンディングボックスやセグメンテーションマスク用のインタラクティブなオーバーレイ付き
    model(
        [
            "./assets/img1.jpeg",
            "./assets/img3.png",
            "./assets/img4.jpeg",
            "./assets/img5.jpeg",
        ]
    )
```

トレーニングやファインチューニングのワークフローでは、`wandb.init()` で run を明示的に初期化する必要はありません。ただし、コードが予測のみを行う場合は、明示的に run を作成する必要があります。

インタラクティブな bbox オーバーレイは次のように表示されます:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

詳しくは [W&B 画像オーバーレイ ガイド]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ja" >}}) を参照してください。

## More resources

* [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)