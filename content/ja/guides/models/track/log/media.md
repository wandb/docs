---
title: メディアやオブジェクトをログする
description: 3Dポイントクラウドや分子、HTMLやヒストグラムなど、さまざまなリッチメディアをログできます
menu:
  default:
    identifier: ja-guides-models-track-log-media
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb" >}}

画像、動画、音声など様々なメディアに対応しています。リッチなメディアをログして結果を探索し、Run、モデル、データセットを視覚的に比較しましょう。以下に例やハウツーガイドを紹介します。

{{% alert %}}
詳細は[データ型リファレンス]({{< relref path="/ref/python/sdk/data-types/" lang="ja" >}})をご覧ください。
{{% /alert %}}

{{% alert %}}
さらに詳しい情報は、[モデル予測の可視化に関するデモレポート](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)や[動画ガイド](https://www.youtube.com/watch?v=96MxRvx15Ts)もご覧ください。
{{% /alert %}}

## 前提条件
W&B SDK でメディアオブジェクトをログするには、追加の依存パッケージのインストールが必要な場合があります。  
以下のコマンドを実行してインストールできます。

```bash
pip install wandb[media]
```

## 画像

画像をログすると入力・出力・フィルタの重み・活性化などを記録できます。

{{< img src="/images/track/log_images.png" alt="オートエンコーダーの入力と出力" >}}

画像は NumPy 配列、PIL イメージ、またはファイルシステムから直接ログできます。

各ステップごとに画像をログすると、UI に保存されます。画像パネルを展開し、ステップスライダーを使って異なるステップの画像を確認できます。これにより、トレーニング中にモデルの出力がどのように変化するかの比較も簡単です。

{{% alert %}}
トレーニングのボトルネックや結果閲覧時の画像読み込み遅延を防ぐため、1ステップあたり50枚未満の画像のログを推奨します。
{{% /alert %}}

{{< tabpane text=true >}}
   {{% tab header="配列を画像としてログする" %}}
配列を直接渡して画像を手動で作成できます。例えば [`torchvision` の `make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid) を利用できます。

配列は [Pillow](https://pillow.readthedocs.io/en/stable/index.html) によって png 変換されます。

```python
import wandb

with wandb.init(project="image-log-example") as run:

    images = wandb.Image(image_array, caption="上：出力、下：入力")

    run.log({"examples": images})
```

配列の最後の次元が1ならグレースケール画像、3なら RGB、4なら RGBA とみなします。浮動小数の場合は `0` から `255` の整数に変換します。独自の正規化方法を適用したい場合 [`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) を直接指定したり、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を渡すこともできます（「PIL 画像をログする」タブ参照）。
   {{% /tab %}}
   {{% tab header="PIL Images をログする" %}}
配列から画像への変換を細かく制御したい場合は、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を自分で作成して直接渡します。

```python
from PIL import Image

with wandb.init(project="") as run:
    # NumPy 配列から PIL 画像を作成
    image = Image.fromarray(image_array)

    # 必要なら RGB に変換
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 画像のログ
    run.log({"example": wandb.Image(image, caption="My Image")})
```

   {{% /tab %}}
   {{% tab header="ファイルから画像をログする" %}}
さらに柔軟にコントロールしたい場合は、画像を任意に作成し、ディスクに保存し、そのファイルパスを渡します。

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


## 画像のオーバーレイ

{{< tabpane text=true >}}
   {{% tab header="セマンティックセグメンテーションマスク" %}}
セマンティックセグメンテーションのマスク（領域分割マスク）をログし、W&B UI で重ね合わせ表示や透過度調整、時間変化の確認などができます。

{{< img src="/images/track/semantic_segmentation.gif" alt="インタラクティブなマスク表示" >}}

オーバーレイをログするには、`wandb.Image` の `masks` 引数に以下の Key-Value を含む辞書を渡します。

* マスク画像を表す2つのキーのどちらか：
  * `"mask_data"`: 2D NumPy 配列、各ピクセルのラベルを整数で指定
  * `"path"`: （文字列）保存済みマスク画像のファイルパス
* `"class_labels"`: （オプション）画像マスク内のラベル整数とクラス名を対応させる辞書

複数のマスクをログしたい場合は、下記コードスニペットのように複数 Key を含めて辞書を作成します。

[ライブ例を見る](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[サンプルコードはこちら](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix_P4J)

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

各ステップ（`run.log()` の呼び出しごと）でマスクがキーごとに定義されます。
- 同じキーに対して異なる値がステップで与えられた場合は最新の値のみが画像に適用されます。
- ステップごとに別のキーが提供された場合、それぞれのキーごとの値すべてが UI に表示されますが、そのステップで定義されたもののみ画像に適用されます。定義されていないマスクの公開範囲を切り替えても画像自体は変化しません。
   {{% /tab %}}
    {{% tab header="バウンディングボックス" %}}
画像とともにバウンディングボックスをログし、UI のフィルターやトグルを使ってさまざまなボックス群の可視化ができます。

{{< img src="/images/track/bb-docs.jpeg" alt="バウンディングボックスの例" >}}

[ライブ例を見る](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログするには、`wandb.Image` の `boxes` 引数に以下の Key-Value 構造の辞書を渡してください。

* `box_data`: 辞書のリスト。各バウンディングボックスごとに1つの辞書。ボックス辞書のフォーマットは以下の通りです。
  * `position`: ボックスの位置とサイズを表す辞書で、下記2パターンいずれか。各ボックスで異なる形式でも可。
    * _オプション1:_ `{"minX", "maxX", "minY", "maxY"}` 各次元の上下限座標を指定
    * _オプション2:_ `{"middle", "width", "height"}` `middle` は `[x, y]` の座標リスト, `width`・`height` はスカラー
  * `class_id`: 各ボックスのクラスを表す整数（`class_labels` キー参照）
  * `scores`: スコアのラベルと値の辞書。UI でボックスをフィルタする際に使用可能
  * `domain`: ボックス座標の単位/形式を指定。**画素単位の場合は "pixel" を指定してください。** デフォルトは画像サイズに対する割合（0〜1 の浮動小数点数）です。
  * `box_caption`: （オプション）このボックス用のラベル表示文字列
* `class_labels`: （オプション） `class_id` とクラス名の対応辞書。指定しなければ `class_0`、`class_1` 等を自動生成します。

例はこちら：

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
                    # デフォルトの割合（相対値）ドメインでのボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセルドメインでのボックス例
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 必要な数だけボックスを追加
                }
            ],
            "class_labels": class_id_to_label,
        },
        # グループごとにユニークなキー名でログ
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


## テーブル内での画像オーバーレイ

{{< tabpane text=true >}}
   {{% tab header="セマンティックセグメンテーションマスク" %}}
{{< img src="/images/track/Segmentation_Masks.gif" alt="テーブル内のインタラクティブなセマンティックマスク" >}}

テーブル内でセマンティックセグメンテーションマスクをログするには各行で `wandb.Image` オブジェクトを作成してください。

下記がコード例です。

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
   {{% tab header="バウンディングボックス" %}}
{{< img src="/images/track/Bounding_Boxes.gif" alt="テーブル内のインタラクティブなバウンディングボックス" >}}

テーブル内でバウンディングボックス付き画像をログする場合も、各行ごとに `wandb.Image` オブジェクトを作成します。

コード例は以下の通りです。

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
リストや配列、テンソルなどの数値列を最初の引数に渡すと、自動で `np.histogram` によりヒストグラムが作成されます。すべての配列・テンソルは1次元化されます。デフォルトは `64` ビンですが、`num_bins` キーワード引数で変更できます（最大 `512` ビンまで）。

UI 上では、x軸が学習ステップ、y軸が指標値、色で頻度を表し、トレーニング中の変化比較が容易です。単発のヒストグラムログ例はこのパネルの「サマリー内ヒストグラム」タブ参照。

```python
run.log({"gradients": wandb.Histogram(grads)})
```

{{< img src="/images/track/histograms.png" alt="GAN 識別器の勾配" >}}   
   {{% /tab %}}
   {{% tab header="柔軟なヒストグラムのログ" %}}
より詳細な制御をしたい場合は、`np.histogram` の返すタプルを `np_histogram` 引数に渡せます。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
run.log({"gradients": wandb.Histogram(np_hist_grads)})
```
   {{% /tab %}}
{{< /tabpane >}}


ヒストグラムが summary に含まれていれば [Run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の Overview タブに表示されます。history にある場合は Charts タブでヒートマップとして時系列表示されます。

## 3D 可視化

3D ポイントクラウドや Lidar シーン（バウンディングボックス付）をログできます。NumPy 配列で各ポイントの座標と色を指定して渡します。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

run.log({"point_cloud": wandb.Object3D(point_cloud)})
```

{{% alert %}}
W&B UI では最大 300,000 点まで表示されます。
{{% /alert %}}

#### NumPy 配列のフォーマット

カラースキームに柔軟に対応した3種の NumPy 配列フォーマットをサポートします。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c はカテゴリー` `[1, 14]` の範囲（セグメンテーション用途）
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は `[0,255]` の赤・緑・青チャンネル値

#### Python オブジェクト

このスキーマを使い、Python オブジェクトを定義して [ `from_point_cloud` メソッド]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_point_cloud" lang="ja" >}}) に渡せます。

* `points` ... 上記「シンプルなポイントクラウド表示」同様の形式の NumPy 配列
* `boxes` ... 3 属性を持つ python 辞書の NumPy 配列
  * `corners` - 8個のコーナーリスト
  * `label` - ボックスに表示する文字列（オプション）
  * `color` - RGB 値
  * `score` - バウンディングボックス表示用の数値（例: `score` が `0.75` 以上のみ表示等）（オプション）
* `type` - シーン種別を表す文字列。現在サポートされている値は `"lidar/beta"` のみ

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
    # ... 以下略
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
         "color": [0, 0, 255], # バウンディングボックスの色（RGB）
         "label": "car", # ボックスに表示される文字列
         "score": 0.6 # バウンディングボックス上に表示される数値
     }],
     vectors = [
        {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5], "color": [255, 0, 0]}, # 色はオプション
     ],
     point_cloud_type = "lidar/beta",
)})
```

ポイントクラウドの閲覧時には control キーを押しながらマウスで空間を自在に移動できます。

#### ポイントクラウドのファイル

[ `from_file` メソッド]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_file" lang="ja" >}}) を使い JSON ファイルからポイントクラウドデータを読み込めます。

```python
run.log({"my_cloud_from_file": wandb.Object3D.from_file(
     "./my_point_cloud.pts.json"
)})
```

フォーマット例：

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

[上記配列フォーマット]({{< relref path="#numpy-array-formats" lang="ja" >}})のいずれかで、[ `from_numpy` メソッド]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_numpy" lang="ja" >}}) で直接ポイントクラウドを定義できます。

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
run.log({"protein": wandb.Molecule("6lu7.pdb")})
```

分子データ（`pdb`、`pqr`、`mmcif`、`mcif`、`cif`、`sdf`、`sd`、`gro`、`mol2`、`mmtf` の計10種類）をログできます。

SMILES 文字列、[`rdkit`](https://www.rdkit.org/docs/index.html) の `mol` ファイル、`rdkit.Chem.rdchem.Mol` オブジェクトのログにも対応しています。

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

Run 完了後、UI 上で分子の3D可視化インタラクションが可能です。

[AlphaFold を使ったライブ例を見る](https://wandb.me/alphafold-workspace)

{{< img src="/images/track/docs-molecule.png" alt="分子構造" >}}
  </TabItem>
</Tabs>

### PNG画像

[`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}})は、`numpy` 配列や `PILImage` インスタンスをデフォルトで PNG 形式に変換します。

```python
run.log({"example": wandb.Image(...)})
# 複数画像の例
run.log({"example": [wandb.Image(...) for img in images]})
```

### 動画

動画は [`wandb.Video`]({{< relref path="/ref/python/sdk/data-types/Video" lang="ja" >}}) データ型でログできます。

```python
run.log({"example": wandb.Video("myvideo.mp4")})
```

これでメディアブラウザで動画を閲覧可能です。プロジェクトワークスペースやRunのワークスペース、レポートで **Add visualization** をクリックしてリッチメディアパネルとして追加しましょう。

## 分子の2Dビュー

[`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) データ型と [`rdkit`](https://www.rdkit.org/docs/index.html) を使い、分子の2D画像をログできます。

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

run.log({"acetic_acid": wandb.Image(pil_image)})
```

## その他メディア

W&B は他にも様々なメディア型をサポートしています。

### 音声

```python
run.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

1ステップあたり最大100クリップまで音声ログ可能です。詳細は[`audio-file`]({{< relref path="/ref/query-panel/audio-file.md" lang="ja" >}}) をご覧ください。

### 動画

```python
run.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

NumPy 配列を渡した場合は「時間・チャンネル・幅・高さ」の順に次元を仮定します。デフォルトは4fpsの gif 画像（`ffmpeg` と [`moviepy`](https://pypi.org/project/moviepy/) が必要）を作成。 `"gif"`、`"mp4"`、`"webm"`、`"ogg"` をサポートします。文字列パスの場合は存在チェックと形式チェック後、アップロードします。`BytesIO` オブジェクトなら指定フォーマットの拡張子で一時ファイルを作成します。

W&B の [Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) や [Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) ページで動画は Media セクションに表示されます。

詳細は[`video-file`]({{< relref path="/ref/query-panel/video-file" lang="ja" >}})を参照ください。

### テキスト

W&B でテキストを表としてログしたい場合は `wandb.Table` を活用してください。デフォルトのカラム名は `["Input", "Output", "Expected"]` です。UIパフォーマンス維持のため、デフォルト最大行数は10,000ですが、`wandb.Table.MAX_ROWS = {DESIRED_MAX}` で上限を変更できます。

```python
with wandb.init(project="my_project") as run:
    columns = ["Text", "Predicted Sentiment", "True Sentiment"]
    # 方法1
    data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
    table = wandb.Table(data=data, columns=columns)
    run.log({"examples": table})

    # 方法2
    table = wandb.Table(columns=columns)
    table.add_data("I love my phone", "1", "1")
    table.add_data("My phone sucks", "0", "-1")
    run.log({"examples": table})
```

pandas の `DataFrame` オブジェクトもそのまま渡せます。

```python
table = wandb.Table(dataframe=my_dataframe)
```

詳細は[`string`]({{< relref path="/ref/query-panel/" lang="ja" >}})をご覧ください。

### HTML

```python
run.log({"custom_file": wandb.Html(open("some.html"))})
run.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

任意のキーで HTML をログでき、Run ページに HTML パネルが表示されます。 デフォルトでスタイルも自動付与されますが、`inject=False` で無効化可能です。

```python
run.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

詳細は[`html-file`]({{< relref path="/ref/query-panel/html-file" lang="ja" >}})を参照ください。