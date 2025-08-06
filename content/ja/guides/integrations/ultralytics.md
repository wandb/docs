---
title: Ultralytics
menu:
  default:
    identifier: ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、ポーズ推定など、最先端のコンピュータビジョンモデルが揃うリポジトリです。[YOLOv8](https://docs.ultralytics.com/models/yolov8/)（リアルタイムオブジェクト検出モデルの最新バージョン）だけでなく、[SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)、[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) などの強力なモデルも取り扱っています。これらのモデルの実装だけでなく、Ultralytics ではトレーニングやファインチューニング、推論を簡単な API で使えるエンドツーエンドのワークフローも提供されています。

## はじめに

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

    開発チームでは、`ultralyticsv8.0.238` 以下のバージョンにてインテグレーションの検証を行っています。インテグレーションに関する不具合は、タグ `yolov8` を付けて [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## 実験管理とバリデーション結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使ったトレーニング、ファインチューニング、バリデーションの一般的なワークフローと、実験管理・モデルのチェックポイント保存・パフォーマンス可視化など [W&B](https://wandb.ai/site) を活用する方法をご紹介します。

統合については、こちらのレポートもご覧ください：[Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics で W&B を使うには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

お好きな `YOLO` モデルを初期化し、推論前に `add_wandb_callback` を呼び出します。これにより、トレーニング・ファインチューニング・バリデーション・推論時に、実験ログや画像（アノテーション情報入り）、各種予測結果が [W&B のインタラクティブなコンピュータビジョン用オーバーレイ]({{< relref "/guides/models/track/log/media#image-overlays-in-tables" >}}) として保存され、[`wandb.Table`]({{< relref "/guides/models/tables/" >}}) に追加情報付きで記録されます。

```python
with wandb.init(project="ultralytics", job_type="train") as run:

    # YOLO モデルの初期化
    model = YOLO("yolov8n.pt")

    # Ultralytics 用の W&B コールバックを追加
    add_wandb_callback(model, enable_model_checkpointing=True)

    # モデルのトレーニングまたはファインチューニングを実施
    # 各エポックの終了時に、バリデーションバッチの予測が
    # W&B の Table にログされ、インタラクティブな可視化が行われます
    model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)
```

Ultralytics のトレーニングやファインチューニングのワークフローで、W&B の実験管理を利用した場合の例です。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

エポックごとのバリデーション結果が [W&B Table]({{< relref "/guides/models/tables/" >}}) で可視化された例はこちらです。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルによる推論ワークフローおよび [W&B](https://wandb.ai/site) を使った結果の可視化方法を紹介します。

Google Colab でコードを試すこともできます：[Colab で開く](https://wandb.me/ultralytics-inference)

統合については、こちらのレポートもご覧ください：[Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

W&B と Ultralytics を統合するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートしましょう。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

統合テスト用に画像をダウンロードします。静止画はもちろん、動画やカメラソースも利用できます。推論ソースについて詳しくは [Ultralytics のドキュメント](https://docs.ultralytics.com/modes/predict/) をご覧ください。

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

`wandb.init()` を使って W&B の [run]({{< relref "/guides/models/track/runs/" >}}) を初期化します。その後、お好きな `YOLO` モデルを初期化し、推論の前に `add_wandb_callback` を呼び出してください。推論時に、[インタラクティブなコンピュータビジョン用オーバーレイ]({{< relref "/guides/models/track/log/media#image-overlays-in-tables" >}}) や追加情報が入った [`wandb.Table`]({{< relref "/guides/models/tables/" >}}) に、画像が自動ログされます。

```python
# W&B Run の初期化
with wandb.init(project="ultralytics", job_type="inference") as run:
    # YOLO モデルの初期化
    model = YOLO("yolov8n.pt")

    # Ultralytics 用の W&B コールバックを追加
    add_wandb_callback(model, enable_model_checkpointing=True)

    # 推論を実行すると、W&B Table へ自動でログ化されます
    # バウンディングボックスやセグメンテーションマスクのインタラクティブな可視化付きです
    model(
        [
            "./assets/img1.jpeg",
            "./assets/img3.png",
            "./assets/img4.jpeg",
            "./assets/img5.jpeg",
        ]
    )
```

トレーニングまたはファインチューニングの場合は、`wandb.init()` で run を明示的に初期化する必要はありません。推論だけの場合は、必ず run を作成してください。

インタラクティブな bbox オーバーレイの例はこちらです。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

詳しくは [W&B 画像オーバーレイ ガイド]({{< relref "/guides/models/track/log/media.md#image-overlays" >}}) をご覧ください。

## その他のリソース

* [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)