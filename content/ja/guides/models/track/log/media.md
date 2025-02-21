---
title: Log media and objects
description: 3D ポイントクラウド や分子から HTML やヒストグラムまで、リッチメディアを ログ
menu:
  default:
    identifier: ja-guides-models-track-log-media
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb" >}}

画像、ビデオ、オーディオなど、様々な形式のメディアをサポートしています。リッチメディアを記録して、結果を分析し、run、model、およびデータセットを視覚的に比較できます。以下の例とハウツー ガイドをご覧ください。

{{% alert %}}
メディアタイプのリファレンスドキュメントをお探しですか？[こちらのページ]({{< relref path="/ref/python/data-types/" lang="ja" >}})をご覧ください。
{{% /alert %}}

{{% alert %}}
[wandb.ai での結果の表示](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)や、[ビデオチュートリアルの視聴](https://www.youtube.com/watch?v=96MxRvx15Ts)が可能です。
{{% /alert %}}

## 前提条件
W&B SDK でメディアオブジェクトを記録するには、追加の依存関係をインストールする必要がある場合があります。
次のコマンドを実行して、これらの依存関係をインストールできます。

```bash
pip install wandb[media]
```

## 画像

画像を記録して、入力、出力、フィルタの重み、アクティベーションなどを追跡します。

{{< img src="/images/track/log_images.png" alt="Inputs and outputs of an autoencoder network performing in-painting." >}}

画像は、NumPy 配列から直接、PIL 画像として、またはファイルシステムから記録できます。

ステップから画像を記録するたびに、UI に表示するために保存されます。画像 パネルを展開し、ステップスライダーを使用して、さまざまなステップの画像を確認します。これにより、トレーニング中にmodelの出力がどのように変化するかを簡単に比較できます。

{{% alert %}}
トレーニング中のログ記録がボトルネックになることや、結果を表示する際の画像読み込みがボトルネックになることを防ぐため、ステップあたり 50 枚未満の画像をログに記録することをお勧めします。
{{% /alert %}}

{{< tabpane text=true >}}
   {{% tab header="配列を画像として記録" %}}
[`torchvision`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)の [`make_grid` の使用などにより、手動で画像を作成する場合は、配列を直接指定します。

配列は、[Pillow](https://pillow.readthedocs.io/en/stable/index.html)を使用して png に変換されます。

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

最後の次元が 1 の場合はグレースケール画像、3 の場合は RGB、4 の場合は RGBA であると想定されます。配列に浮動小数点が含まれている場合は、`0` から `255` の間の整数に変換します。画像を別の方法で正規化する場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)を手動で指定するか、この パネルの「PIL 画像の記録」タブで説明されているように、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)を指定します。
   {{% /tab %}}
   {{% tab header="PIL 画像の記録" %}}
配列から画像への変換を完全に制御するには、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)を自分で作成し、直接指定します。

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```   
   {{% /tab %}}
   {{% tab header="ファイルから画像を記録" %}}
さらに細かく制御するには、好きな方法で画像を作成し、ディスクに保存して、ファイルパスを指定します。

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```   
   {{% /tab %}}
{{< /tabpane >}}


## 画像オーバーレイ


{{< tabpane text=true >}}
   {{% tab header="セグメンテーションマスク" %}}
セマンティックセグメンテーションマスクを記録し、W&B UI を介してそれら（不透明度の変更、経時的な変化の表示など）を操作します。

{{< img src="/images/track/semantic_segmentation.gif" alt="Interactive mask viewing in the W&B UI." >}}

オーバーレイを記録するには、`wandb.Image` の `masks` キーワード引数に、次のキーと値を持つ辞書を指定する必要があります。

* 画像マスクを表す 2 つのキーのいずれか 1 つ。
  * `"mask_data"`: 各ピクセルの整数クラスラベルを含む 2D NumPy 配列
  * `"path"`: (文字列) 保存された画像マスクファイルへのパス
* `"class_labels"`: (オプション) 画像マスク内の整数クラスラベルを読み取り可能なクラス名にマッピングする辞書

複数のマスクを記録するには、以下のコードスニペットのように、複数のキーを持つマスク辞書を記録します。

[ライブの例を見る](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

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
   {{% /tab %}}
    {{% tab header="バウンディングボックス" %}}
画像とバウンディングボックスを記録し、フィルタとトグルを使用して、UI でさまざまなボックスセットを動的に可視化します。

{{< img src="/images/track/bb-docs.jpeg" alt="" >}}

[ライブの例を見る](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスを記録するには、`wandb.Image` の boxes キーワード引数に、次のキーと値を持つ辞書を指定する必要があります。

* `box_data`: 各ボックスに対して 1 つずつ、辞書のリスト。ボックス辞書の形式については、以下で説明します。
  * `position`: ボックスの位置とサイズを表す辞書。以下で説明する 2 つの形式のいずれかになります。ボックスはすべて同じ形式を使用する必要はありません。
    * _オプション 1:_ `{"minX", "maxX", "minY", "maxY"}`。各ボックスの次元の上限と下限を定義する座標セットを指定します。
    * _オプション 2:_ `{"middle", "width", "height"}`。`middle` 座標を `[x,y]` として、`width` と `height` をスカラーとして指定する座標セットを指定します。
  * `class_id`: ボックスのクラス ID を表す整数。以下の `class_labels` キーを参照してください。
  * `scores`: スコアの文字列ラベルと数値の値の辞書。UI でのボックスのフィルタリングに使用できます。
  * `domain`: ボックス座標の単位/形式を指定します。ボックス座標がピクセル空間（画像の次元の範囲内の整数など）で表される場合は、**これを「pixel」に設定してください**。デフォルトでは、ドメインは画像の分数/パーセント（0 ～ 1 の間の浮動小数点数として表される）であると想定されます。
  * `box_caption`: (オプション) このボックスのラベルテキストとして表示される文字列
* `class_labels`: (オプション) `class_id` を文字列にマッピングする辞書。デフォルトでは、クラスラベル `class_0`、`class_1` などが生成されます。

次の例を確認してください。

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
                    # one box expressed in the default relative/fractional domain
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # another box expressed in the pixel domain
                    # (for illustration purposes only, all boxes are likely
                    # to be in the same domain/format)
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # Log as many boxes an as needed
                }
            ],
            "class_labels": class_id_to_label,
        },
        # Log each meaningful group of boxes with a unique key name
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```    
    {{% /tab %}}
{{< /tabpane >}}



## テーブル内の画像オーバーレイ

{{< tabpane text=true >}}
   {{% tab header="セグメンテーションマスク" %}}
{{< img src="/images/track/Segmentation_Masks.gif" alt="Interactive Segmentation Masks in Tables" >}}

テーブルにセグメンテーションマスクを記録するには、テーブル内の各行に対して `wandb.Image` オブジェクトを指定する必要があります。

コードスニペットに例を示します。

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
   {{% /tab %}}
   {{% tab header="バウンディングボックス" %}}
{{< img src="/images/track/Bounding_Boxes.gif" alt="Interactive Bounding Boxes in Tables" >}}

テーブルにバウンディングボックスを含む画像を記録するには、テーブル内の各行に対して `wandb.Image` オブジェクトを指定する必要があります。

コードスニペットに例を示します。

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
   {{% tab header="基本的なヒストグラムの記録" %}}
リスト、配列、テンソルなど、数値のシーケンスが最初の引数として指定されている場合、`np.histogram` を呼び出すことによってヒストグラムが自動的に構築されます。すべての配列/テンソルはフラット化されます。オプションの `num_bins` キーワード引数を使用して、デフォルトの `64` ビンをオーバーライドできます。サポートされているビンの最大数は `512` です。

UI では、ヒストグラムは x 軸にトレーニングステップ、y 軸にメトリック値、色で表されるカウントでプロットされ、トレーニング全体で記録されたヒストグラムの比較が容易になります。1 回限りのヒストグラムの記録の詳細については、このパネルの「概要のヒストグラム」タブを参照してください。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

{{< img src="/images/track/histograms.png" alt="Gradients for the discriminator in a GAN." >}}   
   {{% /tab %}}
   {{% tab header="柔軟なヒストグラムの記録" %}}
さらに細かく制御する場合は、`np.histogram` を呼び出して、返されたタプルを `np_histogram` キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # if only in summary, only visible on overview tab
    {"final_logits": wandb.Histogram(logits)}
)
```   
   {{% /tab %}}
   {{% tab header="概要のヒストグラム" %}}

`'obj'、'gltf'、'glb'、'babylon'、'stl'、'pts.json'` 形式でファイルを記録すると、runが完了したときに UI でレンダリングされます。

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

{{< img src="/images/track/ground_truth_prediction_of_3d_point_clouds.png" alt="Ground truth and prediction of a headphones point cloud" >}}

[ライブの例を見る](https://app.wandb.ai/nbaryd/SparseConvNet-examples_3d_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)   
   {{% /tab %}}
{{< /tabpane >}}



ヒストグラムが概要にある場合、[Run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の Overview タブに表示されます。履歴にある場合は、Charts タブに経時的なビンのヒートマップがプロットされます。

## 3D 可視化


  </TabItem>
  <TabItem value="point_clouds">

バウンディングボックスを含む 3D ポイントクラウドと Lidar シーンを記録します。レンダリングする点の座標と色を含む NumPy 配列を渡します。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

:::info
W&B UI は、データを 300,000 ポイントで切り捨てます。
:::

#### NumPy 配列形式

柔軟なカラースキームを実現するために、3 つの異なる形式の NumPy 配列がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c はカテゴリ` `[1, 14]` の範囲 (セグメンテーションに役立ちます)
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は赤、緑、青のカラー チャンネルの `[0,255]` の範囲の値です。

#### Python オブジェクト

このスキーマを使用すると、Python オブジェクトを定義し、[the `from_point_cloud` method]({{< relref path="/ref/python/data-types/object3d/#from_point_cloud" lang="ja" >}})に以下に示すように渡すことができます。

* `points` は、[上記の単純なポイントクラウド レンダラーと同じ形式を使用して]({{< relref path="#python-object" lang="ja" >}})レンダリングする点の座標と色を含む NumPy 配列です。
* `boxes` は、3 つの属性を持つ Python 辞書の NumPy 配列です。
  * `corners`- 8 つの角のリスト
  * `label`- ボックスにレンダリングされるラベルを表す文字列 (オプション)
  * `color`- ボックスの色を表す rgb 値
  * `score` - バウンディングボックスに表示される数値で、表示されるバウンディングボックスをフィルタリングするために使用できます (たとえば、`score` > `0.75` のバウンディングボックスのみを表示するなど)。(オプション)
* `type` は、レンダリングするシーンタイプを表す文字列です。現在サポートされている値は `lidar/beta` のみです。

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
         "color": [0, 0, 255], # color in RGB of the bounding box
         "label": "car", # string displayed on the bounding box
         "score": 0.6 # numeric displayed on the bounding box
     }],
     vectors = [
        {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5], "color": [255, 0, 0]}, # color is optional
     ],
     point_cloud_type = "lidar/beta",
)})
```

ポイントクラウドを表示するときは、Ctrl キーを押しながらマウスを使用すると、空間内を移動できます。

#### ポイントクラウドファイル

[`from_file` メソッド]({{< relref path="/ref/python/data-types/object3d/#from_file" lang="ja" >}})を使用すると、ポイントクラウドデータでいっぱいの JSON ファイルにロードできます。

```python
run.log({"my_cloud_from_file": wandb.Object3D.from_file(
     "./my_point_cloud.pts.json"
)})
```

ポイントクラウドデータの形式設定方法の例を以下に示します。

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

[上記で定義した同じ配列形式]({{< relref path="#numpy-array-formats" lang="ja" >}})を使用すると、[`from_numpy` メソッド]({{< relref path="/ref/python/data-types/object3d/#from_numpy" lang="ja" >}})で `numpy` 配列を直接使用して、ポイントクラウドを定義できます。

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
wandb.log({"protein": wandb.Molecule("6lu7.pdb")})
```

10 個のファイルタイプのいずれかで分子データを記録します: `pdb`、`pqr`、`mmcif`、`mcif`、`cif`、`sdf`、`sd`、`gro`、`mol2`、または `mmtf`。

W&B は、SMILES 文字列、[`rdkit`](https://www.rdkit.org/docs/index.html) `mol` ファイル、および `rdkit.Chem.rdchem.Mol` オブジェクトからの分子データの記録もサポートしています。

```python
resveratrol = rdkit.Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")

wandb.log(
    {
        "resveratrol": wandb.Molecule.from_rdkit(resveratrol),
        "green fluorescent protein": wandb.Molecule.from_rdkit("2b3p.mol"),
        "acetaminophen": wandb.Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1"),
    }
)
```

runが完了すると、UI で分子の 3D 可視化を操作できるようになります。

[AlphaFold を使用したライブの例を見る](http://wandb.me/alphafold-workspace)

{{< img src="/images/track/docs-molecule.png" alt="" >}}
  </TabItem>
</Tabs>

### PNG 画像

[`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}})は、デフォルトで `numpy` 配列または `PILImage` のインスタンスを PNG に変換します。

```python
wandb.log({"example": wandb.Image(...)})
# Or multiple images
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### ビデオ

ビデオは、[`wandb.Video`]({{< relref path="/ref/python/data-types/video.md" lang="ja" >}})データ型を使用して記録されます。

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

これで、メディアブラウザでビデオを表示できます。プロジェクト ワークスペース、run ワークスペース、または report に移動し、[**可視化を追加**] をクリックして、リッチメディア パネルを追加します。

## 分子の 2D ビュー

[`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}})データ型と[`rdkit`](https://www.rdkit.org/docs/index.html)を使用して、分子の 2D ビューを記録できます。

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```


## その他のメディア

W&B は、さまざまな他のメディアタイプの記録もサポートしています。

### オーディオ

```python
wandb.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

ステップごとに最大 100 個のオーディオクリップを記録できます。詳細な使用方法については、[`audio-file`]({{< relref path="/ref/query-panel/audio-file.md" lang="ja" >}})を参照してください。

### ビデオ

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy 配列が指定されている場合、次元は時間、チャンネル、幅、高さの順であると想定されます。デフォルトでは、4 fps の gif 画像が作成されます ([`ffmpeg`](https://www.ffmpeg.org) および [`moviepy`](https://pypi.org/project/moviepy/) python ライブラリは、numpy オブジェクトを渡す場合に必要です)。サポートされている形式は、`"gif"`、`"mp4"`、`"webm"`、および `"ogg"` です。文字列を `wandb.Video` に渡すと、wandb にアップロードする前に、ファイルが存在し、サポートされている形式であることをアサートします。`BytesIO` オブジェクトを渡すと、指定された形式が拡張子として指定された一時ファイルが作成されます。

W&B の [Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})および [Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) ページでは、メディアセクションにビデオが表示されます。

詳細な使用方法については、[`video-file`]({{< relref path="/ref/query-panel/video-file" lang="ja" >}})を参照してください。

### テキスト

UI に表示するために、`wandb.Table` を使用してテーブルにテキストを記録します。デフォルトでは、列ヘッダーは `["Input", "Output", "Expected"]` です。最適な UI パフォーマンスを確保するために、デフォルトの最大行数は 10,000 に設定されています。ただし、ユーザーは `wandb.Table.MAX_ROWS = {DESIRED_MAX}` を使用して最大値を明示的にオーバーライドできます。

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# Method 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# Method 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

pandas `DataFrame` オブジェクトを渡すこともできます。

```python
table = wandb.Table(dataframe=my_dataframe)
```

詳細な使用方法については、[`string`]({{< relref path="/ref/query-panel/" lang="ja" >}})を参照してください。

### HTML

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

カスタム HTML は任意のキーで記録でき、これにより、run ページに HTML パネルが表示されます。デフォルトでは、デフォルトのスタイルが挿入されます。`inject=False` を渡すことで、デフォルトのスタイルをオフにすることができます。

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

詳細な使用方法については、[`html-file`]({{< relref path="/ref/query-panel/html-file" lang="ja" >}})を参照してください。
