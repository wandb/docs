---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Ultralytics

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

[Ultralytics](https://github.com/ultralytics/ultralytics)は、画像分類、オブジェクト検出、画像セグメンテーション、ポーズ推定などのタスクのための最先端のコンピュータビジョンモデルを提供します。ここでは、リアルタイムオブジェクト検出モデルのYOLOシリーズの最新バージョンである[YOLOv8](https://docs.ultralytics.com/models/yolov8/)を始め、[SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/)、[YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/)などの強力なコンピュータビジョンモデルもホストしています。さらに、これらのモデルの実装だけでなく、トレーニング、ファインチューニング、適用を簡単に行えるAPIも提供しています。

## はじめに

まず、`ultralytics`と`wandb`をインストールします。

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install --upgrade ultralytics==8.0.238 wandb

# または
# conda install ultralytics
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install --upgrade ultralytics==8.0.238 wandb
```

  </TabItem>
</Tabs>

**注意:** 現在のインテグレーションは`ultralyticsv8.0.238`およびそれ以前のバージョンでテストされています。問題があればhttps://github.com/wandb/wandb/issuesに`yolov8`タグを付けて報告してください。

## 実験管理と検証結果の可視化

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

このセクションでは、Ultralyticsモデルを使用したトレーニング、ファインチューニング、検証、および実験管理、モデルのチェックポイント、パフォーマンスの可視化を行う典型的なワークフローを示します。

Google Colabでコードを試してみることもできます：[Colabで開く](http://wandb.me/ultralytics-train)

インテグレーションについての詳細はこのレポートを参照してください：[Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

W&BとUltralyticsを連携させるためには、`wandb.integration.ultralytics.add_wandb_callback`関数をインポートする必要があります。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

次に、お好みの`YOLO`モデルを初期化し、推論を行う前に`add_wandb_callback`関数を呼び出します。これにより、トレーニング、ファインチューニング、検証、または推論を行う際に、実験ログや予測結果を自動的にログし、W&Bの[コンピュータビジョンタスク用のインタラクティブなオーバーレイ](../track/log/media#image-overlays-in-tables)および追加のインサイトが含まれた[`wandb.Table`](../tables/intro.md)として可視化されます。

```python
# YOLOモデルを初期化
model = YOLO("yolov8n.pt")

# Ultralytics用のW&Bコールバックを追加
add_wandb_callback(model, enable_model_checkpointing=True)

# モデルをトレーニング/ファインチューニング
# 各エポックの終わりに、検証バッチの予測がW&Bテーブルに
# インタラクティブなオーバーレイと共にログされます
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# W&Bのrunを終了
wandb.finish()
```

こちらは、W&Bを使用してUltralyticsのトレーニングまたはファインチューニングワークフローを管理した例です:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

こちらは、エポックごとの検証結果をW&Bテーブルで可視化した例です:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 予測結果の可視化

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb"></CTAButtons>

このセクションでは、Ultralyticsモデルを使用して推論を行い、その結果をW&Bを使用して可視化する典型的なワークフローを示します。

Google Colabでコードを試してみることもできます：[Colabで開く](http://wandb.me/ultralytics-inference)。

インテグレーションについての詳細はこのレポートを参照してください：[Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

W&BとUltralyticsを連携させるためには、`wandb.integration.ultralytics.add_wandb_callback`関数をインポートする必要があります。

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

次に、いくつかの画像をダウンロードしてインテグレーションをテストします。自分の画像、動画、カメラソースを使用することも可能です。推論ソースについての詳細は[公式ドキュメント](https://docs.ultralytics.com/modes/predict/)を参照してください。

```python
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

次に、`wandb.init`を使ってW&B [run](../runs/intro.md)を初期化します。

```python
# W&Bのrunを初期化
wandb.init(project="ultralytics", job_type="inference")
```

次に、お好みの`YOLO`モデルを初期化し、推論を行う前に`add_wandb_callback`関数を呼び出します。これにより、推論を行う際に自動的に画像がログされ、W&Bの[コンピュータビジョンタスク用のインタラクティブなオーバーレイ](../track/log/media#image-overlays-in-tables)および追加のインサイトが含まれた[`wandb.Table`](../tables/intro.md)として可視化されます。

```python
# YOLOモデルを初期化
model = YOLO("yolov8n.pt")

# Ultralytics用のW&Bコールバックを追加
add_wandb_callback(model, enable_model_checkpointing=True)

# 推論を実行し、結果をインタラクティブなオーバーレイ付きでW&Bテーブルに自動的にログ
model(["./assets/img1.jpeg", "./assets/img3.png", "./assets/img4.jpeg", "./assets/img5.jpeg"])

# W&Bのrunを終了
wandb.finish()
```

注意: トレーニングやファインチューニングのワークフローの場合、`wandb.init()`を明示的に初期化する必要はありませんが、コードが推論のみを含む場合、runを明示的に作成する必要があります。

こちらはインタラクティブなバウンディングボックスオーバーレイの例です:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&Bの画像オーバーレイの詳細は[こちら](../track/log/media.md#image-overlays)を参照してください。

## その他のリソース

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)