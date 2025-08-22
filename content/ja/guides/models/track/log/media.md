---
title: メディアとオブジェクトをログする
description: 3D ポイント クラウドや分子から HTML、ヒストグラムまで、豊富なメディアをログする
menu:
  default:
    identifier: ja-guides-models-track-log-media
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb" >}}

私たちは画像、ビデオ、音声などをサポートしています。リッチメディアをログして、結果を探索し、Run、Models、Datasetsを視覚的に比較しましょう。例やハウツーガイドは以下をご覧ください。

{{% alert %}}
メディアタイプの参考ドキュメントをお探しですか？この[ページ]({{< relref path="/ref/python/data-types/" lang="ja" >}})が必要です。
{{% /alert %}}

{{% alert %}}
[wandb.ai で結果を確認する](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)ことができ、[ビデオチュートリアルに従う](https://www.youtube.com/watch?v=96MxRvx15Ts)ことができます。
{{% /alert %}}

## 前提条件
W&B SDKを使用してメディアオブジェクトをログするためには、追加の依存関係をインストールする必要があるかもしれません。以下のコマンドを実行してこれらの依存関係をインストールできます：

```bash
pip install wandb[media]
```

## 画像

画像をログして、入力、出力、フィルター重み、活性化状態などを追跡しましょう。

{{< img src="/images/track/log_images.png" alt="ペインティングを行うオートエンコーダーネットワークの入力と出力。" >}}

画像はNumPy配列、PIL画像、またはファイルシステムから直接ログできます。

ステップごとに画像をログするたびに、UIに表示するために保存されます。画像パネルを拡大し、ステップスライダーを使用して異なるステップの画像を確認します。これにより、トレーニング中にモデルの出力がどのように変化するかを比較しやすくなります。

{{% alert %}}
トレーニング中のログのボトルネックを防ぎ、結果を表示する際の画像読み込みのボトルネックを防ぐために、1ステップあたり50枚以下の画像をログすることをお勧めします。
{{% /alert %}}

{{< tabpane text=true >}}
   {{% tab header="配列を画像としてログする" %}}
配列を手動で画像として構築する際に、[`make_grid` from `torchvision`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)を使用するなど、配列を直接提供します。

配列は[Pillow](https://pillow.readthedocs.io/en/stable/index.html)を使用してpngに変換されます。

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

最後の次元が1の場合はグレースケール、3の場合はRGB、4の場合はRGBAと仮定します。配列が浮動小数点数を含む場合、正規化アルゴリズムを使用して自動的に`0`から`255`の整数に変換します。PyTorchテンソルとNumPy配列での正規化の動作についての詳細は、[Imageリファレンスの画像正規化セクション]({{< relref "/ref/python/data-types/image.md#image-normalization" >}})を参照してください。異なる方法で画像を正規화したい場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)を手動で指定するか、`"Logging PIL Images"`タブで説明されているように、単に[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)を提供することができます。
   {{% /tab %}}
   {{% tab header="PIL Imagesをログする" %}}
配列から画像への変換を完全に制御するために、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)を自分で構築し、直接提供してください。

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```   
   {{% /tab %}}
   {{% tab header="ファイルから画像をログする" %}}
さらに制御したい場合は、任意の方法で画像を作成し、ディスクに保存し、ファイルパスを提供します。

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
セマンティックセグメンテーションマスクをログし、W&B UIを通じて（不透明度の変更、時間経過による変化の確認など）それらと対話します。

{{< img src="/images/track/semantic_segmentation.gif" alt="W&B UIでのインタラクティブなマスク表示。" >}}

オーバーレイをログするには、`wandb.Image`の`masks`キーワード引数に以下のキーと値を持つ辞書を提供する必要があります：

* 画像マスクを表す2つのキーのうちの1つ：
  * `"mask_data"`：各ピクセルの整数クラスラベルを含む2D NumPy配列
  * `"path"`：（文字列）保存された画像マスクファイルへのパス
* `"class_labels"`：（オプション）画像マスク内の整数クラスラベルを可読クラス名にマッピングする辞書

複数のマスクをログするには、以下のコードスニペットのように、複数のキーを含むマスク辞書をログします。

[ライブ例を参照してください](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

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
画像にバウンディングボックスをログし、UIで異なるセットのボックスを動的に可視化するためにフィルターや切り替えを使用します。

{{< img src="/images/track/bb-docs.jpeg" alt="" >}}

[ライブ例を参照してください](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログするには、`wandb.Image`の`boxes`キーワード引数に以下のキーと値を持つ辞書を提供する必要があります：

* `box_data`：各ボックス用の辞書リスト。ボックス辞書形式は以下に説明します。
  * `position`：ボックスの位置とサイズを表す辞書で、以下で説明する2つの形式のいずれか。すべてのボックスが同じ形式を使用する必要はありません。
    * _オプション 1:_ `{"minX", "maxX", "minY", "maxY"}`。各ボックスの次元の上下限を定義する座標セットを提供します。
    * _オプション 2:_ `{"middle", "width", "height"}`。`middle`座標を`[x,y]`として、`width`と`height`をスカラーとして指定します。
  * `class_id`：ボックスのクラス識別を表す整数。以下の`class_labels`キーを参照。
  * `scores`：スコアの文字列ラベルと数値の辞書。UIでボックスをフィルタリングするために使用できます。
  * `domain`：ボックス座標の単位/形式を指定してください。**この値を"pixel"に設定**してください。ボックス座標が画像の次元内の整数のようにピクセル空間で表されている場合、デフォルトで、domainは画像の割合/百分率として表され、0から1までの浮動小数点数として解釈されます。
  * `box_caption`：（オプション）このボックス上に表示されるラベルテキストとしての文字列
* `class_labels`：（オプション）`class_id`を文字列にマッピングする辞書。デフォルトでは`class_0`、`class_1`などのクラスラベルを生成します。

この例をチェックしてください：

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
                    # デフォルトの相対/小数領域で表現された1つのボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセル領域で表現された別のボックス
                    # （説明目的のみ、すべてのボックスは同じ領域/形式である可能性が高い）
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 必要に応じて多くのボックスをログします
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 意味のあるボックスのグループごとに一意のキーネームでログします
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
{{< img src="/images/track/Segmentation_Masks.gif" alt="テーブル内のインタラクティブなセグメンテーションマスク" >}}

テーブル内でセグメンテーションマスクをログするには、テーブルの各行に対して`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例があります：

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
{{< img src="/images/track/Bounding_Boxes.gif" alt="テーブル内のインタラクティブなバウンディングボックス" >}}

テーブル内でバウンディングボックス付き画像をログするには、テーブルの各行に`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例があります：

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
   {{% tab header="基本ヒストグラムログ" %}}
リスト、配列、テンソルなどの数字のシーケンスが最初の引数として提供されると、自動的に`np.histogram`を呼んでヒストグラムを構築します。すべての配列/テンソルはフラット化されます。`num_bins`キーワード引数を使用して`64`ビンのデフォルト設定を上書きできます。最大サポートビン数は`512`です。

UIでは、トレーニングステップがx軸に、メトリック値がy軸に、色で表現されたカウントでヒストグラムがプロットされ、トレーニング中にログされたヒストグラムを比較しやすくしています。詳細については、このパネルの"Summary内のヒストグラム"タブを参照してください。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

{{< img src="/images/track/histograms.png" alt="GANのディスクリミネータの勾配。" >}}   
   {{% /tab %}}
   {{% tab header="柔軟なヒストグラムログ" %}}
もっと制御したい場合は、`np.histogram`を呼び出し、その返されたタプルを`np_histogram`キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # Summaryにのみある場合、Overviewタブにのみ表示されます
    {"final_logits": wandb.Histogram(logits)}
)
```   
   {{% /tab %}}
   {{% tab header="Summary内のヒストグラム" %}}

ファイルを形式 `'obj'`, `'gltf'`, `'glb'`, `'babylon'`, `'stl'`, `'pts.json'` でログすれば、runが終了した際にUIでそれらをレンダリングします。

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

{{< img src="/images/track/ground_truth_prediction_of_3d_point_clouds.png" alt="ヘッドホンのポイントクラウドの正解と予測" >}}

[ライブ例を見る](https://app.wandb.ai/nbaryd/SparseConvNet-examples_3d_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)   
   {{% /tab %}}
{{< /tabpane >}}



Summary内にあるヒストグラムは、[Run Page]({{< relref path="/guides/models/track/runs/" lang="ja" >}})のOverviewタブに表示されます。履歴にある場合、Chartsタブで時間経過によるビンのヒートマップをプロットします。

## 3D可視化


  </TabItem>
  <TabItem value="point_clouds">

3Dポイントクラウドとバウンディングボックスを持つLidarシーンをログします。レンダリングするポイントの座標と色を含むNumPy配列を渡します。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

:::info
W&B UIはデータを30万ポイントに制限します。
:::

#### NumPy配列フォーマット

色のスキームに柔軟性を持たせるために、3つの異なるデータ形式のNumPy配列がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4`, `cは範囲[1, 14]内のカテゴリ` （セグメンテーションに便利）
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は赤、緑、青のカラー チャネルに対して範囲`[0,255]`の値

#### Pythonオブジェクト

このスキーマを使用して、Pythonオブジェクトを定義し、以下に示すように [the `from_point_cloud` method]({{< relref path="/ref/python/data-types/object3d/#from_point_cloud" lang="ja" >}}) に渡すことができます。

* `points`は、[単純なポイントクラウドレンダラーで上記に示されたのと同じフォーマットを使用してレンダリングするポイントの座標と色を含むNumPy配列です]({{< relref path="#python-object" lang="ja" >}})。
* `boxes`は3つの属性を持つPython辞書のNumPy配列です：
  * `corners` - 8つのコーナーのリスト
  * `label` - ボックスにレンダリングされるラベルを表す文字列 (オプション)
  * `color` - ボックスの色を表すrgb値
  * `score` - バウンディングボックスに表示される数値で、表示するバウンディングボックスのフィルタリングに使用できます（例：`score` > `0.75`のバウンディングボックスのみを表示する）。(オプション)
* `type`はレンダリングされるシーンタイプを表す文字列です。現在サポートされている値は`lidar/beta`のみです。

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
         "color": [0, 0, 255], # バウンディングボックスのRGB色
         "label": "car", # バウンディングボックスに表示される文字列
         "score": 0.6 # バウンディングボックスに表示される数値
     }],
     vectors = [
        {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5], "color": [255, 0, 0]}, # 色は任意
     ],
     point_cloud_type = "lidar/beta",
)})
```

ポイントクラウドを表示するとき、Controlキーを押しながらマウスを使用すると、内部空間を移動できます。

#### ポイントクラウドファイル

[the `from_file` method]({{< relref path="/ref/python/data-types/object3d/#from_file" lang="ja" >}}) を使用して、ポイントクラウドデータが満載のJSONファイルをロードできます。

```python
run.log({"my_cloud_from_file": wandb.Object3D.from_file(
     "./my_point_cloud.pts.json"
)})
```

ポイントクラウドデータのフォーマット方法の例を以下に示します。

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
#### NumPy配列

[上記で定義された配列フォーマット]({{< relref path="#numpy-array-formats" lang="ja" >}})を使用して、`numpy`配列を直接 [the `from_numpy` method]({{< relref path="/ref/python/data-types/object3d/#from_numpy" lang="ja" >}}) でポイントクラウドを定義できます。

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
            [0.4, 1, 1.3, 1], # x, y, z, カテゴリ 
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

分子データは`pdb`、`pqr`、`mmcif`、`mcif`、`cif`、`sdf`、`sd`、`gro`、`mol2`、`mmtf.`のいずれかの10のファイル形式でログできます。

W&Bはまた、SMILES文字列、[`rdkit`](https://www.rdkit.org/docs/index.html)の`mol`ファイル、`rdkit.Chem.rdchem.Mol`オブジェクトからの分子データのログをサポートします。

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

runが終了すると、UIで分子の3D可視化と対話できるようになります。

[AlphaFoldを使用したライブ例を見る](http://wandb.me/alphafold-workspace)

{{< img src="/images/track/docs-molecule.png" alt="" >}}
  </TabItem>
</Tabs>

### PNG 画像

`wandb.Image`は`numpy`配列や`PILImage`のインスタンスをデフォルトでPNGに変換します。

```python
wandb.log({"example": wandb.Image(...)})
# または複数の画像
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### ビデオ

ビデオは[`wandb.Video`]({{< relref path="/ref/python/data-types/video.md" lang="ja" >}}) データ型を使用してログします：

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

現在、メディアブラウザでビデオを見ることができます。 プロジェクトワークスペース、runワークスペース、またはレポートに移動し、**Add visualization** をクリックしてリッチメディアパネルを追加します。

## 分子の2Dビュー

[`wandb.Image`]({{< relref "/ref/python/data-types/image.md" >}})データ型と[`rdkit`](https://www.rdkit.org/docs/index.html)を使用して分子の2Dビューをログできます:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```


## その他のメディア

W&Bは、さまざまな他のメディアタイプのログもサポートしています。

### オーディオ

```python
wandb.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

1ステップあたりの最大100のオーディオクリップをログできます。詳細な使い方については、[`audio-file`]({{< relref path="/ref/query-panel/audio-file.md" lang="ja" >}})を参照してください。

### ビデオ

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy配列が供給された場合、時間、チャンネル、幅、高さの順であると仮定します。 デフォルトでは4fpsのgif画像を作成します（numpyオブジェクトを渡す場合、[`ffmpeg`](https://www.ffmpeg.org)と[`moviepy`](https://pypi.org/project/moviepy/) pythonライブラリが必要です）。 サポートされているフォーマットは、`"gif"`、`"mp4"`、`"webm"`、そして`"ogg"`です。`wandb.Video` に文字列を渡すと、ファイルが存在し、wandbにアップロードする前にサポートされたフォーマットであることを確認します。 `BytesIO`オブジェクトを渡すと、指定されたフォーマットを拡張子とする一時ファイルを作成します。

W&Bの[Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})と[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}})ページで、メディアセクションにビデオが表示されます。

詳細な使い方については、[`video-file`]({{< relref path="/ref/query-panel/video-file" lang="ja" >}})を参照してください。

### テキスト

`wandb.Table`を使用して、UIに表示するテーブルにテキストをログします。デフォルトで、列ヘッダーは`["Input", "Output", "Expected"]`です。最適なUIパフォーマンスを確保するために、デフォルトで最大行数は10,000に設定されています。ただし、ユーザーは`wandb.Table.MAX_ROWS = {DESIRED_MAX}`を使用して明示的に上限を超えることができます。

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# メソッド 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# メソッド 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

また、pandas `DataFrame`オブジェクトを渡すこともできます。

```python
table = wandb.Table(dataframe=my_dataframe)
```

詳細な使い方については、[`string`]({{< relref path="/ref/query-panel/" lang="ja" >}})を参照してください。

### HTML

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

カスタムHTMLは任意のキーでログ可能で、runページ上にHTMLパネルを接続します。デフォルトではスタイルが注入されます。デフォルトスタイルをオフにするには、`inject=False`を渡します。

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

詳細な使い方については、[`html-file`]({{< relref path="/ref/query-panel/html-file" lang="ja" >}})を参照してください。