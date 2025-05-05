---
title: ウルトラリティクス
menu:
  default:
    identifier: ja-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、ポーズ推定などのタスクにおける最先端のコンピュータビジョンモデルのホームです。リアルタイムオブジェクト検出モデルのYOLOシリーズの最新バージョンである [YOLOv8](https://docs.ultralytics.com/models/yolov8/) をホストするだけでなく、他にも [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)、[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) などの強力なコンピュータビジョンモデルも備えています。これらのモデルの実装を提供するだけでなく、Ultralytics は、これらのモデルを使ったトレーニング、ファインチューニング、適用のための使いやすい API を使ったエンドツーエンドのワークフローも提供しています。

## 始めましょう

1. `ultralytics` と `wandb` をインストールします。

    {{< tabpane text=true >}}
    {{% tab header="Command Line" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # または
    # conda install ultralytics
    ```

    {{% /tab %}}
    {{% tab header="Notebook" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    開発チームは `ultralyticsv8.0.238` 以下とのインテグレーションをテストしました。インテグレーションに関する問題を報告するには、タグ `yolov8` を付けて [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## 実験管理とバリデーション結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使ったトレーニング、ファインチューニング、バリデーションの典型的なワークフローと、実験管理、モデルのチェックポイント、モデルのパフォーマンスの可視化を [W&B](https://wandb.ai/site) を使用して行う方法を示します。

このインテグレーションについては、次のレポートで確認することもできます：[Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics と W&B のインテグレーションを使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

選択した `YOLO` モデルを初期化し、推論を行う前に `add_wandb_callback` 関数を呼び出します。これにより、トレーニング、ファインチューニング、バリデーション、または推論を行うときに、実験ログと、地上真実とそれぞれの予測結果を重ね合わせた画像が、自動的に [コンピュータビジョンタスクの対話型オーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) で保存され、追加の洞察が [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) に保存されることを保証します。

```python
# YOLO モデルを初期化
model = YOLO("yolov8n.pt")

# Ultralytics 用に W&B コールバックを追加
add_wandb_callback(model, enable_model_checkpointing=True)

# モデルをトレーニング/ファインチューニング
# 各エポックの終わりに、バリデーションバッチでの予測が
# コンピュータビジョンタスク用の洞察に満ちた対話型オーバーレイと共に
# W&B テーブルに記録されます
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# W&B run を終了
wandb.finish()
```

Ultralytics のトレーニングまたはファインチューニングワークフローで W&B により実験管理された様子は次のとおりです。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

エポックごとのバリデーション結果が [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) を使用してどのように可視化されるかは次のとおりです。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果の可視化

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使った推論と結果の可視化の典型的なワークフローを [W&B](https://wandb.ai/site) を使用して示します。

Google Colab でコードを試すことができます: [Open in Colab](http://wandb.me/ultralytics-inference).

このインテグレーションについては、次のレポートで確認することもできます：[Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics と W&B のインテグレーションを使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートする必要があります。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

インテグレーションをテストするためにいくつかの画像をダウンロードします。静止画像、ビデオ、またはカメラソースを使用できます。推論ソースの詳細については、[Ultralytics のドキュメント](https://docs.ultralytics.com/modes/predict/) を確認してください。

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

次に、`wandb.init` を使って W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化します。

```python
# W&B run を初期化
wandb.init(project="ultralytics", job_type="inference")
```

次に、希望する `YOLO` モデルを初期化し、推論を行う前に `add_wandb_callback` 関数を呼び出します。これにより、推論を実行すると、[コンピュータビジョンタスク用の対話型オーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) で画像が自動的にログに記録され、追加の洞察が [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) に提供されることを保証します。

```python
# YOLO モデルを初期化
model = YOLO("yolov8n.pt")

# Ultralytics 用に W&B コールバックを追加
add_wandb_callback(model, enable_model_checkpointing=True)

# 予測を実行し、自動的に W&B テーブルにログを記録
# バウンディングボックス、セグメンテーションマスク用の対話型オーバーレイ付き
model(
    [
        "./assets/img1.jpeg",
        "./assets/img3.png",
        "./assets/img4.jpeg",
        "./assets/img5.jpeg",
    ]
)

# W&B run を終了
wandb.finish()
```

トレーニングまたはファインチューニングワークフローの場合、`wandb.init()` を使用して明示的に run を初期化する必要はありません。ただし、コードが予測のみを含む場合は、明示的に run を作成する必要があります。

対話型の bbox オーバーレイは次のように表示されます。

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B 画像オーバーレイに関する詳細情報は [こちら]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ja" >}}) で取得できます。

## その他のリソース

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)