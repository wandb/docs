---
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# YOLOv5

[UltralyticsのYOLOv5](https://ultralytics.com/yolov5)（"You Only Look Once"）モデルファミリーでは、痛みを伴わずに畳み込みニューラルネットワークを使用したリアルタイムのオブジェクト検出が可能です。

[Weights & Biases](http://wandb.com)は、YOLOv5に直接統合されており、実験メトリクスのトラッキング、モデルおよびデータセットのバージョニング、豊富なモデル予測の可視化などを提供します。 **YOLO実験を実行する前に、`pip install`を実行するだけで簡単に使えます！**

:::info
YOLOv5の統合におけるモデルおよびデータログ機能の概要は、[このColab](https://wandb.me/yolo-colab)と下にリンクされたビデオチュートリアルをご覧ください。
:::

<!-- {% embed url="https://www.youtube.com/watch?v=yyecuhBmLxE" %} -->

:::info
すべてのW&Bログ機能は、データ並列マルチGPUトレーニング（例：[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)）と互換性があります。
:::

## コア実験トラッキング

`wandb`をインストールするだけで、組み込まれたW&B[ログ機能](../track/log/intro.md)がアクティブ化されます：システムメトリクス、モデルメトリクス、およびメディアがインタラクティブな[ダッシュボード](../track/app.md)にログされます。

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 小さいデータセットで小さいネットワークを訓練する
```

wandbが標準出力に印刷したリンクをたどるだけで終わりです。
![すべてのこれらのチャートが揃った！](/images/integrations/yolov5_experiment_tracking.png)

## モデルバージョニングとデータ可視化

しかし、それだけではありません！YOLOにいくつかのシンプルなコマンドライン引数を渡すことで、さらに多くのW&B機能を利用できます。

* `--save_period`に数値を渡すと、[モデルのバージョン管理](../model_registry/intro.md)がオンになります。`save_period`エポックの終わりに、モデルの重みがW&Bに保存されます。検証セットで最も性能の良いモデルが自動的にタグ付けされます。
* `--upload_dataset`フラグをオンにすると、データのバージョン管理のためにデータセットもアップロードされます。
* `--bbox_interval`に数値を渡すと、[データ可視化](../tables/intro.md)がオンになります。`bbox_interval`エポックの終わりに、検証セットでのモデルの出力がW&Bにアップロードされます。

<Tabs
  defaultValue="modelversioning"
  values={[
    {label: 'モデルバージョニングのみ', value: 'modelversioning'},
    {label: 'モデルバージョニングとデータ可視化', value: 'bothversioning'},
  ]}>
  <TabItem value="modelversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1
```

  </TabItem>
  <TabItem value="bothversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```
</TabItem>

</Tabs>

:::info

すべてのW&Bアカウントには、データセットとモデル用の100GBの無料ストレージが付属しています。

:::

これがどのように見えるかをご覧ください。

![Model Versioning: 最新および最高のバージョンのモデルが識別されます。](/images/integrations/yolov5_model_versioning.png)

![Data Visualization: モデルの出力と個々のメトリクスとの比較で入力画像を確認します。](/images/integrations/yolov5_data_visualization.png)

:::info

データとモデルのバージョン管理により、一時停止またはクラッシュした実験を任意のデバイスから再開できます。設定は不要です！詳細については、[Colab](https://wandb.me/yolo-colab)をご覧ください。

:::