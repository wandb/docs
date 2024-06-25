---
description: 3Dポイントクラウドや分子からHTMLやヒストグラムまで、リッチメディアをログする
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Log Media & Objects

私たちは画像、ビデオ、オーディオなどをサポートしています。リッチなメディアをログに保存して結果を探求し、Runs、Models、およびDatasetsを視覚的に比較しましょう。以下に例とハウツーガイドを紹介します。

:::info
メディアタイプのリファレンスドキュメントをお探しですか？[このページ](../../../ref/python/data-types/README.md)をご覧ください。
:::

:::info
これらのメディアオブジェクトをログに保存する動作コードは[このColab Notebook](http://wandb.me/media-colab)で確認できます。結果がどう見えるかはwandb.aiの[こちら](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)でチェックし、上記リンクのビデオチュートリアルに従ってください。
:::

## 画像

入力、出力、フィルターウェイト、アクティベーションなどを追跡するために画像をログに保存しましょう！

![自己符号化ネットワークの入力と出力](/images/track/log_images.png)

画像はNumPy配列、PIL画像、またはファイルシステムから直接読み込むことができます。

:::info
トレーニング中のログ保存がボトルネックにならないよう、ステップごとに50枚以下の画像をログに残すことをお勧めします。
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: '配列を画像としてログに保存', value: 'arrays'},
    {label: 'PIL画像をログに保存', value: 'pil_images'},
    {label: 'ファイルから画像をログに保存', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

配列を手動で画像に変換する際に直接提供します。例: [`torchvision の make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid)を使用します。

配列は[Pillow](https://pillow.readthedocs.io/en/stable/index.html)を使用してpngに変換されます。

```python
images = wandb.Image(image_array, caption="上: 出力, 下: 入力")

wandb.log({"examples": images})
```

最後の次元が1の場合はグレースケール、3の場合はRGB、4の場合はRGBAと仮定しています。配列に浮動小数点数が含まれている場合、それを`0`から`255`の間の整数に変換します。異なる正規化を行いたい場合は、自分で[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)を指定するか、[PIL.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html)を提供してください。"PIL画像をログに保存"タブで詳細を説明します。
  </TabItem>
  <TabItem value="pil_images">

配列から画像への変換を完全に制御するために、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)を自分で構築して直接提供します。

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
さらに詳しい制御を行いたい場合は、好きなように画像を作成し、ディスクに保存し、ファイルパスを提供してください。

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## イメージオーバーレイ

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'セグメンテーションマスク', value: 'segmentation_masks'},
    {label: 'バウンディングボックス', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

セマンティックセグメンテーションマスクをログに保存し、W&B UIを通じて不透明度の変更や時間変化の表示などの操作が可能です。

![W&B UIでのインタラクティブなマスク表示](/images/track/semantic_segmentation.gif)

オーバーレイをログに保存するには、`wandb.Image`の`masks`キーワード引数に以下のキーと値を持つ辞書を提供する必要があります。

* イメージマスクを表す2つのキーのうち1つを提供:
  * `"mask_data"`: 各ピクセルの整数クラスラベルを含む2次元のNumPy配列
  * `"path"`: 保存された画像マスクファイルへのパス（文字列）
* `"class_labels"`: （オプション）画像マスク内の整数クラスラベルを読み取り可能なクラス名にマッピングする辞書

複数のマスクをログに保存するには、以下のコードスニペットのように複数のキーを持つマスク辞書をログに保存してください。

[ライブ例を見る →](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[サンプルコード →](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix\_P4J)

```python
mask_data = np.array([[1, 2, 2, ..., 2, 2, 1], ...])

class_labels = {1: "tree", 2: "car", 3: "road"}

mask_img = wandb.Image(
    image,
    masks={
        "predictions": {"mask_data": mask_data, "class_labels": class_labels},
        "ground_truth": {
            # ...
        },
        # ...
    },
)
```
  </TabItem>
  <TabItem value="bounding_boxes">
画像にバウンディングボックスをログに保存し、フィルターやトグルを使用してUIで動的に異なるセットのボックスを視覚化します。

![](@site/static/images/track/bb-docs.jpeg)

[ライブ例を見る →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログに保存するには、以下のキーと値を持つ辞書を`wandb.Image`の`boxes`キーワード引数に提供する必要があります。

* `box_data`: 各ボックスの辞書のリスト。ボックスの辞書フォーマットは以下に記載。
  * `position`: ボックスの位置とサイズを表す辞書。2つのフォーマットのいずれかで提供します。すべてのボックスが同じフォーマットを使用する必要はありません。
    * _オプション 1:_ `{"minX", "maxX", "minY", "maxY"}`. 各ボックス寸法の上下限を定義する座標セットを提供します。
    * _オプション 2:_ `{"middle", "width", "height"}`. 座標`middle`を`[x,y]`として提供し、幅と高さをスカラーで提供します。
  * `class_id`: ボックスのクラスIDを表す整数。`class_labels`キーを参照。
  * `scores`: スコアの文字列ラベルと数値を持つ辞書。UIでボックスをフィルタリングするために使用されます。
  * `domain`: ボックス座標の単位/形式を指定。ボックス座標がピクセル空間で表現されている場合（画像寸法の境界内の整数として）、**これを「pixel」に設定**します。デフォルトでは、domainは画像の割合/パーセンテージ（0から1の間の浮動小数点数）と見なされます。
  * `box_caption`: （オプション）このボックスのラベルテキストとして表示される文字列
* `class_labels`: （オプション）`class_id`を文字列にマッピングする辞書。デフォルトでは`class_0`、`class_1`などのクラスラベルが生成されます。

以下の例をチェックしてください:

```python
class_id_to_label = {
    1: "car",
    2: "road",
    3: "building",
    # ...
}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # デフォルトの相対/割合ドメインで表現された1つのボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセルドメインで表現されたもう1つのボックス
                    # （説明のため、すべてのボックスは同じドメイン/形式である可能性が高い）
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 必要に応じて多くのボックスをログに保存
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 各意味のあるボックスのグループを固有のキー名でログに保存
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## テーブル内の画像オーバーレイ

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'セグメンテーションマスク', value: 'segmentation_masks'},
    {label: 'バウンディングボックス', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![テーブル内のインタラクティブなセグメンテーションマスク](/images/track/Segmentation_Masks.gif)

テーブル内のセグメンテーションマスクをログに保存するには、テーブルの各行に対して`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例を示します:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, label in zip(ids, images, labels):
    mask_img = wandb.Image(
        img,
        masks={
            "prediction": {"mask_data": label, "class_labels": class_labels}
            # ...
        },
    )

    table.add_data(id, img)

wandb.log({"Table": table})
```
  </TabItem>
  <TabItem value="bounding_boxes">

![テーブル内のインタラクティブなバウンディングボックス](/images/track/Bounding_Boxes.gif)

テーブル内のバウンディングボックス付き画像をログに保存するには、テーブルの各行に対して`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例を示します:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, boxes in zip(ids, images, boxes_set):
    box_img = wandb.Image(
        img,
        boxes={
            "prediction": {
                "box_data": [
                    {
                        "position": {
                            "minX": box["minX"],
                            "minY": box["minY"],
                            "maxX": box["maxX"],
                            "maxY": box["maxY"],
                        },
                        "class_id": box["class_id"],
                        "box_caption": box["caption"],
                        "domain": "pixel",
                    }
                    for box in boxes
                ],
                "class_labels": class_labels,
            }
        },
    )
```
  </TabItem>
</Tabs>

## ヒストグラム

<Tabs
  defaultValue="histogram_logging"
  values={[
    {label: '基本的なヒストグラムのログ', value: 'histogram_logging'},
    {label: '柔軟なヒストグラムのログ', value: 'flexible_histogram'},
    {label: 'サマリーでのヒストグラム', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">

数値のシーケンス（例：リスト、配列、テンソル）が最初の引数として提供される場合、`np.histogram`を自動的に呼び出してヒストグラムを構築します。すべての配列/テンソルがフラット化されることに注意してください。オプションの`num_bins`キーワード引数を使用して、デフォルトの`64`ビンを上書きすることができます。サポートされる最大ビン数は`512`です。

UIでは、トレーニングステップがx軸、メトリック値がy軸、カウントが色で表されます。これにより、トレーニング中にログに保存されたヒストグラムを比較しやすくなります。サマリーでのヒストグラムのログの詳細については、このパネルの「サマリーでのヒストグラム」タブを参照してください。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GANの判別器の勾配](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

より詳細な制御を行いたい場合は、`np.histogram`を呼び出し、返されたタプルを`np_histogram`キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # サマリー内にのみ表示され、Overviewタブでのみ表示されます。
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

ヒストグラムがサマリーにある場合、[Runページ](../../app/pages/run-page.md)のOverviewタブに表示されます。ヒストグラムがヒストリーにある場合、Chartsタブでビンのヒートマップが時間とともにプロットされます。

## 3D 可視化

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3Dオブジェクト', value: '3d_object'},
    {label: 'ポイントクラウド', value: 'point_clouds'},
    {label: '分子', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

`'obj'`, `'gltf'`, `'glb'`, `'babylon'`, `'stl'`, `'pts.json'`形式のファイルをログに保存し、Runが終了した時にUIでレンダリングします。

```python
wandb.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

![ヘッドホンのポイントクラウドの正解と予測](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[ライブ例を見る →](https://app.wandb.ai/nbaryd/SparseConvNet-examples\_3d\_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

3Dポイントクラウドおよびライダースケーンをログに保存し、バウンディングボックスを追加します。レンダリングするポイントの座標と色を含むNumPy配列を渡します。UIでは、300,000ポイントに切り捨てます。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

柔軟なカラースキームに対応する3つの異なる形状のNumPy配列がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| cはカテゴリー` `[1から14の範囲]`（セグメンテーションに便利）
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` `[0,255の範囲]`で赤、緑、青の各チャネルの値

以下のログコードの例を示します:

* `points`は単純なポイントクラウドレンダラーと同じ形式のNumPy配列です。
* `boxes`は次の3つの属性を持つpython辞書のNumPy配列です:
  * `corners` -