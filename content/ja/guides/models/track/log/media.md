---
title: メディアとオブジェクトのログ
description: 3D 点群や分子から HTML やヒストグラムまで、リッチメディアをログ
menu:
  default:
    identifier: ja-guides-models-track-log-media
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb" >}}

画像、動画、オーディオなどをサポートしています。リッチメディアをログして 結果を探索し、Runs、Models、Datasets を視覚的に比較しましょう。以下でサンプルや手順の ガイド を紹介します。

{{% alert %}}
詳しくは、[Data types reference]({{< relref path="/ref/python/sdk/data-types/" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{% alert %}}
さらに詳しく知りたい場合は、[モデルの予測を可視化する デモ Report](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA) をチェックするか、[動画ガイド](https://www.youtube.com/watch?v=96MxRvx15Ts) をご覧ください。
{{% /alert %}}

## 前提条件
W&B SDK でメディア オブジェクトをログするには、追加の依存関係が必要になる場合があります。
次のコマンドでインストールできます:

```bash
pip install wandb[media]
```

## 画像

画像をログして、入力、出力、フィルターの重み、活性化などを追跡します。

{{< img src="/images/track/log_images.png" alt="オートエンコーダーの 入力と出力" >}}

画像は NumPy 配列、PIL 画像、またはファイルシステムから直接ログできます。

各ステップで画像をログするたびに、UI に表示できるよう保存します。画像 パネルを展開し、ステップ スライダーを使って異なるステップの画像を確認できます。これにより、トレーニング中に Model の出力がどのように変化するかを簡単に比較できます。

{{% alert %}}
トレーニング時のログ処理や、結果閲覧時の画像読み込みがボトルネックになるのを防ぐため、1 ステップあたり 50 枚未満の画像をログすることを推奨します。
{{% /alert %}}

{{< tabpane text=true >}}
   {{% tab header="配列を画像としてログする" %}}
`torchvision` の [`make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid) などを使って手動で画像を作る際、配列をそのまま渡せます。

配列は [Pillow](https://pillow.readthedocs.io/en/stable/index.html) によって PNG に変換されます。

```python
import wandb

with wandb.init(project="image-log-example") as run:

    images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

    run.log({"examples": images})
```

最後の次元が 1 ならグレースケール、3 なら RGB、4 なら RGBA の画像として扱います。配列が float を含む場合は、`0`〜`255` の整数に変換します。異なる正規化を行いたい場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) を手動で指定するか、このパネルの「PIL 画像をログする」タブで説明しているように [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を直接渡してください。   
   {{% /tab %}}
   {{% tab header="PIL 画像をログする" %}}
配列から画像への変換を細かく制御したい場合は、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を自分で作成して直接渡します。

```python
from PIL import Image

with wandb.init(project="") as run:
    # NumPy 配列から PIL 画像を作成
    image = Image.fromarray(image_array)

    # 必要に応じて RGB に変換
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 画像をログ
    run.log({"example": wandb.Image(image, caption="My Image")})
```

   {{% /tab %}}
   {{% tab header="ファイルから画像をログする" %}}
さらに細かく制御したい場合は、任意の方法で画像を作成してディスクに保存し、ファイルパスを渡します。

```python
import wandb
from PIL import Image

with wandb.init(project="") as run:

    im = Image.fromarray(...)
    rgb_im = im.convert("RGB")
    rgb_im.save("myimage.jpg")

    run.log({"example": wandb.Image("myimage.jpg")})
```   
   {{% /tab %}}
{{< /tabpane >}}


## 画像オーバーレイ


{{< tabpane text=true >}}
   {{% tab header="セマンティックセグメンテーション マスク" %}}
セマンティックセグメンテーション マスクをログし、W&B の UI 上で不透明度の変更、時間変化の確認などの操作ができます。

{{< img src="/images/track/semantic_segmentation.gif" alt="インタラクティブな マスク表示" >}}

オーバーレイをログするには、`wandb.Image` の `masks` キーワード引数に、以下の キー と 値 を持つ 辞書 を渡します:

* 画像マスクを表す次のいずれかのキー
  * `"mask_data"`: 各ピクセルの整数クラスラベルを含む 2D NumPy 配列
  * `"path"`: （文字列）保存済みの画像マスク ファイルへのパス
* `"class_labels"`: （任意）画像マスク内の整数クラスラベルを、人が読めるクラス名に対応付ける 辞書

複数のマスクをログするには、以下の コードスニペット のように、複数のキーを持つマスク 辞書 をログします。

[ライブ例を見る](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[サンプルコード](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix_P4J)

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

各キーに対するセグメンテーション マスクは、各ステップ（`run.log()` の各呼び出し）で定義されます。
- 同じマスク キーに対してステップごとに異なる値を提供した場合、画像に適用されるのはそのキーの最新の値のみです。
- ステップごとに異なるマスク キーを提供した場合、各キーの値はすべて表示されますが、表示中のステップで定義されたマスクのみが画像に適用されます。表示中のステップで定義されていないマスクの表示切り替えは、画像を変更しません。
   {{% /tab %}}
    {{% tab header="バウンディング ボックス" %}}
画像にバウンディング ボックスをログし、フィルターやトグルを使って UI 上で異なるボックス集合を動的に可視化できます。

{{< img src="/images/track/bb-docs.jpeg" alt="バウンディング ボックスの例" >}}

[ライブ例を見る](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディング ボックスをログするには、`wandb.Image` の boxes キーワード引数に、以下の キー と 値 を持つ 辞書 を渡します:

* `box_data`: 各ボックスにつき 1 つの辞書からなるリスト。ボックス辞書の形式は以下のとおりです。
  * `position`: ボックスの位置とサイズを表す辞書。以下のいずれかの形式で指定します。すべてのボックスが同じ形式である必要はありません。
    * オプション 1: `{"minX", "maxX", "minY", "maxY"}`。各次元の上下限となる座標を指定します。
    * オプション 2: `{"middle", "width", "height"}`。`middle` を `[x,y]` の座標で、`width` と `height` をスカラーで指定します。
  * `class_id`: ボックスのクラス ID を表す整数。下記の `class_labels` キーを参照。
  * `scores`: スコア用の文字列ラベルと数値の辞書。UI でのボックスのフィルタリングに使えます。
  * `domain`: ボックス座標の単位/形式を指定します。ボックス座標が画像内のピクセル空間（画像サイズの範囲内の整数など）で表されている場合は、必ず "pixel" を設定してください。デフォルトでは、座標は画像に対する割合（0〜1 の浮動小数点）と見なされます。
  * `box_caption`: （任意）このボックスに表示するラベル文字列
* `class_labels`: （任意）`class_id` を文字列に対応付ける 辞書。指定がない場合は `class_0`、`class_1` のように自動生成します。

例をご覧ください:

```python
import wandb

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
                    # 既定の相対（割合）ドメインで表した 1 つ目のボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセル ドメインで表した別のボックス
                    # （説明のため。実際にはすべてのボックスが
                    # 同じドメイン/形式であることが多い）
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 必要な数だけボックスをログできます
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 意味のあるボックス群ごとに一意のキー名でログ
        "ground_truth": {
            # ...
        },
    },
)

with wandb.init(project="my_project") as run:
    run.log({"driving_scene": img})
```    
    {{% /tab %}}
{{< /tabpane >}}



## テーブルでの 画像オーバーレイ

{{< tabpane text=true >}}
   {{% tab header="セマンティックセグメンテーション マスク" %}}
{{< img src="/images/track/Segmentation_Masks.gif" alt="テーブルでの インタラクティブな セマンティックセグメンテーション マスク" >}}

テーブルで セマンティックセグメンテーション マスクをログするには、テーブルの各行に `wandb.Image` オブジェクトを用意します。

以下の コードスニペット に例を示します:

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

    table.add_data(id, mask_img)

with wandb.init(project="my_project") as run:
    run.log({"Table": table})
```   
   {{% /tab %}}
   {{% tab header="バウンディング ボックス" %}}
{{< img src="/images/track/Bounding_Boxes.gif" alt="テーブルでの インタラクティブな バウンディング ボックス" >}}

テーブルで バウンディング ボックス付きの画像をログするには、テーブルの各行に `wandb.Image` オブジェクトを用意します。

以下の コードスニペット に例を示します:

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
   {{% /tab %}}
{{< /tabpane >}}



## ヒストグラム

{{< tabpane text=true >}}
   {{% tab header="基本的なヒストグラムのログ" %}}
最初の引数として数値列（list、配列、テンソル など）を渡すと、`np.histogram` を呼び出して自動的にヒストグラムを作成します。すべての配列/テンソルはフラット化されます。任意の `num_bins` キーワード引数で既定のビン数 `64` を上書きできます。サポートされる最大ビン数は `512` です。

UI では、ヒストグラムは x 軸にトレーニング ステップ、y 軸にメトリクス値、色でカウントを表し、トレーニング全体でログされたヒストグラムの比較を容易にします。単発のヒストグラムをログする方法は、このパネルの「サマリーのヒストグラム」タブを参照してください。

```python
run.log({"gradients": wandb.Histogram(grads)})
```

{{< img src="/images/track/histograms.png" alt="GAN 識別器の 勾配" >}}   
   {{% /tab %}}
   {{% tab header="柔軟なヒストグラムのログ" %}}
より細かく制御したい場合は、自分で `np.histogram` を呼び、返り値のタプルを `np_histogram` キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
run.log({"gradients": wandb.Histogram(np_hist_grads)})
```
   {{% /tab %}}
{{< /tabpane >}}



ヒストグラムが summary にある場合は、[Run Page]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の Overviewタブ に表示されます。history にある場合は、Charts タブに時間軸でビンのヒートマップを描画します。

## 3D 可視化

3D 点群や Lidar シーンをバウンディング ボックス付きでログできます。レンダリングする点の座標と色を含む NumPy 配列を渡してください。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

run.log({"point_cloud": wandb.Object3D(point_cloud)})
```

{{% alert %}}
W&B の UI は 300,000 点で データを切り捨てます。
{{% /alert %}}

#### NumPy 配列フォーマット

柔軟な配色のために、3 種類の NumPy 配列形式をサポートします。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c はカテゴリ` `[1, 14]` の範囲（セグメンテーションに便利）
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は `[0,255]` の範囲の赤、緑、青の各チャンネル値

#### Python オブジェクト

このスキーマを用いれば、Python オブジェクトを定義し、[`from_point_cloud` メソッド]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_point_cloud" lang="ja" >}}) に渡せます。

* `points` は、レンダリングする点の座標と色を含む NumPy 配列です（[上で示したシンプルな点群レンダラーと同じ形式]({{< relref path="#python-object" lang="ja" >}})）。
* `boxes` は、3 つの属性を持つ Python 辞書の NumPy 配列です:
  * `corners` - 8 つのコーナーのリスト
  * `label` - ボックスに表示するラベル文字列（任意）
  * `color` - ボックスの色を表す rgb 値
  * `score` - バウンディング ボックスに表示され、表示するボックスのフィルタに使える数値（例: `score` > `0.75` のボックスのみを表示）。任意
* `type` はレンダリングするシーンタイプを表す文字列です。現在サポートされている値は `lidar/beta` のみです。

```python
point_list = [
    [
        2566.571924017235, # x
        746.7817289698219, # y
        -15.269245470863748,# z
        76.5, # red
        127.5, # green
        89.46617199365393 # blue
    ],
    [ 2566.592983606823, 746.6791987335685, -15.275803826279521, 76.5, 127.5, 89.45471117247024 ],
    [ 2566.616361739416, 746.4903185513501, -15.28628929674075, 76.5, 127.5, 89.41336375503832 ],
    [ 2561.706014951675, 744.5349468458361, -14.877496818222781, 76.5, 127.5, 82.21868245418283 ],
    [ 2561.5281847916694, 744.2546118233013, -14.867862032341005, 76.5, 127.5, 81.87824684536432 ],
    [ 2561.3693562897465, 744.1804761656741, -14.854129178142523, 76.5, 127.5, 81.64137897587152 ],
    [ 2561.6093071504515, 744.0287526628543, -14.882135189841177, 76.5, 127.5, 81.89871499537098 ],
    # ... and so on
]

run.log({"my_first_point_cloud": wandb.Object3D.from_point_cloud(
     points = point_list,
     boxes = [{
         "corners": [
                [ 2601.2765123137915, 767.5669506323393, -17.816764802288663 ],
                [ 2599.7259021588347, 769.0082337923552, -17.816764802288663 ],
                [ 2599.7259021588347, 769.0082337923552, -19.66876480228866 ],
                [ 2601.2765123137915, 767.5669506323393, -19.66876480228866 ],
                [ 2604.8684867834395, 771.4313904894723, -17.816764802288663 ],
                [ 2603.3178766284827, 772.8726736494882, -17.816764802288663 ],
                [ 2603.3178766284827, 772.8726736494882, -19.66876480228866 ],
                [ 2604.8684867834395, 771.4313904894723, -19.66876480228866 ]
        ],
         "color": [0, 0, 255], # バウンディング ボックスの RGB 色
         "label": "car", # バウンディング ボックスに表示する文字列
         "score": 0.6 # バウンディング ボックスに表示する数値
     }],
     vectors = [
        {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5], "color": [255, 0, 0]}, # color は任意
     ],
     point_cloud_type = "lidar/beta",
)})
```

点群を表示する際は、Ctrl を押しながらマウスで空間内を移動できます。

#### 点群ファイル

[`from_file` メソッド]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_file" lang="ja" >}}) を使って、点群データを含む JSON ファイルを読み込めます。

```python
run.log({"my_cloud_from_file": wandb.Object3D.from_file(
     "./my_point_cloud.pts.json"
)})
```

点群データのフォーマット例は以下のとおりです。

```json
{
    "boxes": [
        {
            "color": [
                0,
                255,
                0
            ],
            "score": 0.35,
            "label": "My label",
            "corners": [
                [
                    2589.695869075582,
                    760.7400443552185,
                    -18.044831294622487
                ],
                [
                    2590.719039645323,
                    762.3871153874499,
                    -18.044831294622487
                ],
                [
                    2590.719039645323,
                    762.3871153874499,
                    -19.54083129462249
                ],
                [
                    2589.695869075582,
                    760.7400443552185,
                    -19.54083129462249
                ],
                [
                    2594.9666662674313,
                    757.4657929961453,
                    -18.044831294622487
                ],
                [
                    2595.9898368371723,
                    759.1128640283766,
                    -18.044831294622487
                ],
                [
                    2595.9898368371723,
                    759.1128640283766,
                    -19.54083129462249
                ],
                [
                    2594.9666662674313,
                    757.4657929961453,
                    -19.54083129462249
                ]
            ]
        }
    ],
    "points": [
        [
            2566.571924017235,
            746.7817289698219,
            -15.269245470863748,
            76.5,
            127.5,
            89.46617199365393
        ],
        [
            2566.592983606823,
            746.6791987335685,
            -15.275803826279521,
            76.5,
            127.5,
            89.45471117247024
        ],
        [
            2566.616361739416,
            746.4903185513501,
            -15.28628929674075,
            76.5,
            127.5,
            89.41336375503832
        ]
    ],
    "type": "lidar/beta"
}
```
#### NumPy 配列

[上で定義したのと同じ配列形式]({{< relref path="#numpy-array-formats" lang="ja" >}})を使い、[`from_numpy` メソッド]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_numpy" lang="ja" >}})で `numpy` 配列から直接点群を定義できます。

```python
run.log({"my_cloud_from_numpy_xyz": wandb.Object3D.from_numpy(
     np.array(  
        [
            [0.4, 1, 1.3], # x, y, z
            [1, 1, 1], 
            [1.2, 1, 1.2]
        ]
    )
)})
```
```python
run.log({"my_cloud_from_numpy_cat": wandb.Object3D.from_numpy(
     np.array(  
        [
            [0.4, 1, 1.3, 1], # x, y, z, category 
            [1, 1, 1, 1], 
            [1.2, 1, 1.2, 12], 
            [1.2, 1, 1.3, 12], 
            [1.2, 1, 1.4, 12], 
            [1.2, 1, 1.5, 12], 
            [1.2, 1, 1.6, 11], 
            [1.2, 1, 1.7, 11], 
        ]
    )
)})
```
```python
run.log({"my_cloud_from_numpy_rgb": wandb.Object3D.from_numpy(
     np.array(  
        [
            [0.4, 1, 1.3, 255, 0, 0], # x, y, z, r, g, b 
            [1, 1, 1, 0, 255, 0], 
            [1.2, 1, 1.3, 0, 255, 255],
            [1.2, 1, 1.4, 0, 255, 255],
            [1.2, 1, 1.5, 0, 0, 255],
            [1.2, 1, 1.1, 0, 0, 255],
            [1.2, 1, 0.9, 0, 0, 255],
        ]
    )
)})
```

  </TabItem>
  <TabItem value="molecules">

```python
run.log({"protein": wandb.Molecule("6lu7.pdb")})
```

分子データは次の 10 種類のファイル形式でログできます: `pdb`、`pqr`、`mmcif`、`mcif`、`cif`、`sdf`、`sd`、`gro`、`mol2`、`mmtf`。

W&B はまた、SMILES 文字列、[`rdkit`](https://www.rdkit.org/docs/index.html) の `mol` ファイル、`rdkit.Chem.rdchem.Mol` オブジェクトからの分子データのログもサポートしています。

```python
resveratrol = rdkit.Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")

run.log(
    {
        "resveratrol": wandb.Molecule.from_rdkit(resveratrol),
        "green fluorescent protein": wandb.Molecule.from_rdkit("2b3p.mol"),
        "acetaminophen": wandb.Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1"),
    }
)
```

run が終了すると、UI 上で分子の 3D 可視化を操作できるようになります。

[AlphaFold を用いたライブ例を見る](https://wandb.me/alphafold-workspace)

{{< img src="/images/track/docs-molecule.png" alt="分子構造" >}}
  </TabItem>
</Tabs>

### PNG 画像

[`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) は、`numpy` 配列や `PILImage` のインスタンスを既定で PNG に変換します。

```python
run.log({"example": wandb.Image(...)})
# 複数画像のログ
run.log({"example": [wandb.Image(...) for img in images]})
```

### 動画

動画は [`wandb.Video`]({{< relref path="/ref/python/sdk/data-types/Video" lang="ja" >}}) データ型でログします:

```python
run.log({"example": wandb.Video("myvideo.mp4")})
```

これでメディア ブラウザーで動画を見られます。project workspace、run workspace、または report に移動し、**Add visualization** をクリックしてリッチメディア パネルを追加します。

## 分子の 2D 表示

[`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) データ型と [`rdkit`](https://www.rdkit.org/docs/index.html) を使って、分子の 2D 表示をログできます:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

run.log({"acetic_acid": wandb.Image(pil_image)})
```


## その他のメディア

W&B は、他のさまざまなメディアタイプのログもサポートしています。

### オーディオ

```python
run.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

1 ステップあたり最大 100 本のオーディオ クリップをログできます。使い方の詳細は [`audio-file`]({{< relref path="/ref/query-panel/audio-file.md" lang="ja" >}}) を参照してください。

### 動画

```python
run.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy 配列を渡した場合、次元の順序は 時間、チャネル、幅、高さ と見なします。既定では 4 fps の GIF を作成します（numpy オブジェクトを渡す場合は、[`ffmpeg`](https://www.ffmpeg.org) と Python ライブラリの [`moviepy`](https://pypi.org/project/moviepy/) が必要です）。サポートされる形式は `"gif"`、`"mp4"`、`"webm"`、`"ogg"` です。文字列を `wandb.Video` に渡した場合は、アップロード前にそのファイルの存在とサポート対象形式であることを確認します。`BytesIO` オブジェクトを渡すと、指定した形式の拡張子を持つ一時ファイルを作成します。

W&B の [Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) ページ と [Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) ページ の Media セクションに動画が表示されます。

使い方の詳細は [`video-file`]({{< relref path="/ref/query-panel/video-file" lang="ja" >}}) を参照してください。

### テキスト

`wandb.Table` を使って テーブル にテキストをログし、UI に表示できます。既定の列ヘッダーは `["Input", "Output", "Expected"]` です。UI パフォーマンス最適化のため、既定の最大行数は 10,000 に設定されていますが、`wandb.Table.MAX_ROWS = {DESIRED_MAX}` で明示的に上書きできます。

```python
with wandb.init(project="my_project") as run:
    columns = ["Text", "Predicted Sentiment", "True Sentiment"]
    # 方法 1
    data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
    table = wandb.Table(data=data, columns=columns)
    run.log({"examples": table})

    # 方法 2
    table = wandb.Table(columns=columns)
    table.add_data("I love my phone", "1", "1")
    table.add_data("My phone sucks", "0", "-1")
    run.log({"examples": table})
```

pandas の `DataFrame` オブジェクトを渡すこともできます。

```python
table = wandb.Table(dataframe=my_dataframe)
```

使い方の詳細は [`string`]({{< relref path="/ref/query-panel/" lang="ja" >}}) を参照してください。

### HTML

```python
run.log({"custom_file": wandb.Html(open("some.html"))})
run.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

任意のキーでカスタム HTML をログでき、Run ページに HTML パネルが表示されます。既定では標準のスタイルを注入します。`inject=False` を渡すと標準スタイルの注入をオフにできます。

```python
run.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

使い方の詳細は [`html-file`]({{< relref path="/ref/query-panel/html-file" lang="ja" >}}) を参照してください。