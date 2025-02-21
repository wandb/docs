---
title: Ultralytics
menu:
  default:
    identifier: ja-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) は、画像分類、オブジェクト検出、画像セグメンテーション、ポーズ推定といったタスクのための最先端コンピュータビジョンモデルのホームです。これは、リアルタイムオブジェクト検出モデルシリーズの最新バージョンである [YOLOv8](https://docs.ultralytics.com/models/yolov8/) だけでなく、他の強力なコンピュータビジョンモデルもホストしています。例えば [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) などです。これらのモデルの実装を提供するだけでなく、Ultralytics は使いやすい API を使用して、これらのモデルをトレーニング、ファインチューン、および適用するための即戦力のワークフローを提供しています。

## 始める

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

    開発チームは `ultralyticsv8.0.238` 以前のバージョンでインテグレーションのテストをしています。インテグレーションに関する問題を報告するには、`yolov8` タグ付きで [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) を作成してください。

## 実験管理を追跡し、検証結果を可視化する

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

このセクションでは、トレーニング、ファインチューニング、検証のために[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使用し、実験管理、モデルチェックポイント、およびモデルのパフォーマンスの可視化を [W&B](https://wandb.ai/site) を使用して行う典型的なワークフローを示します。

このインテグレーションについては、以下のレポートで確認することもできます: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics との W&B インテグレーションを使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートします。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

お好みの `YOLO` モデルを初期化し、推論を行う前に `add_wandb_callback` 関数を呼び出します。これにより、トレーニング、ファインチューニング、検証、または推論を行うときに、実験ログとともに、Ground-Truth およびそれぞれの予測結果を兼ねた画像が自動的に保存されます。このプロセスでは、[コンピュータビジョンタスクのインタラクティブオーバーレイ]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) の画像が W&B に保存され、追加の洞察も[`wandb.Table`]({{< relref path="/guides/core/tables/" lang="ja" >}}) で得られます。

```python
# YOLO モデルを初期化
model = YOLO("yolov8n.pt")

# Ultralytics 用の W&B コールバックを追加
add_wandb_callback(model, enable_model_checkpointing=True)

# モデルをトレーニング/ファインチューン
# 各エポックの終わりに、検証バッチの予測が
# 解説のあるインタラクティブなオーバーレイで
# W&B テーブルにログされます
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# W&B run を終了
wandb.finish()
```

Ultralytics のトレーニングまたはファインチューニングワークフローにおける、W&B を使用した実験管理の追跡結果はこちらです:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO ファインチューニング実験管理</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

エポックごとの検証結果がどのように [W&B Table]({{< relref path="/guides/core/tables/" lang="ja" >}}) で可視化されるかはこちらです:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB 検証可視化テーブル</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果を可視化する

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

このセクションでは、[Ultralytics](https://docs.ultralytics.com/modes/predict/) モデルを使用した推論と、その結果を [W&B](https://wandb.ai/site) を使って可視化する典型的なワークフローを示します。

Google Colab でコードを試してみることができます: [Colabで開く](http://wandb.me/ultralytics-inference)。

このインテグレーションについては、以下のレポートで確認することもできます: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics との W&B インテグレーションを使用するには、`wandb.integration.ultralytics.add_wandb_callback` 関数をインポートする必要があります。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

インテグレーションをテストするためにいくつかの画像をダウンロードします。スチル画像、ビデオ、またはカメラソースを使用できます。推論ソースの詳細については、[Ultralytics のドキュメント](https://docs.ultralytics.com/modes/predict/)を参照してください。

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

次に、`wandb.init` を使って W&B [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を初期化します。

```python
# W&B run を初期化
wandb.init(project="ultralytics", job_type="inference")
```

次に、希望する `YOLO` モデルを初期化し、推論を行う前にその上で `add_wandb_callback` 関数を呼び出します。これにより、推論を実行するたびに、[インタラクティブオーバーレイ付きのコンピュータビジョンタスク]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ja" >}}) に画像が自動的にログされ、追加の洞察が [`wandb.Table`]({{< relref path="/guides/core/tables/" lang="ja" >}}) で得られます。

```python
# YOLO モデルを初期化
model = YOLO("yolov8n.pt")

# Ultralytics 用の W&B コールバックを追加
add_wandb_callback(model, enable_model_checkpointing=True)

# インタラクティブオーバーレイで、境界ボックス、セグメンテーションマスクが
# 自動的にログされる W&B テーブルへの予測を実行
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

トレーニングまたはファインチューンワークフローの場合、`wandb.init()` を使用して run を明示的に初期化する必要はありません。ただし、コードが予測のみを含む場合は、明示的に run を作成する必要があります。

インタラクティブな境界ボックスオーバーレイはこのように見えます:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB イメージオーバーレイ</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B の画像オーバーレイに関する詳細情報は[こちら]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ja" >}})でご覧いただけます。

## その他のリソース

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)