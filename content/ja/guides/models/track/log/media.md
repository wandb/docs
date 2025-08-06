---
title: メディアやオブジェクトをログする
description: 3D ポイントクラウドや分子、HTML、ヒストグラムなどのリッチメディアをログする
menu:
  default:
    identifier: media
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb" >}}

W&B では、画像・動画・音声など多彩なメディアのログに対応しています。リッチメディアを記録し、 結果を分析・可視化したり、Runs や Models、Datasets を見た目で比較しましょう。以降で具体例や手順を紹介します。

{{% alert %}}
詳細は [データ型リファレンス]({{< relref "/ref/python/sdk/data-types/" >}}) を参照してください。
{{% /alert %}}

{{% alert %}}
より詳しい内容は、[モデル予測の可視化デモレポート](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA) や [動画ガイド](https://www.youtube.com/watch?v=96MxRvx15Ts) をご覧ください。
{{% /alert %}}

## 事前準備
W&B SDK でメディアオブジェクトをログするには追加の依存ライブラリが必要な場合があります。次のコマンドでインストールできます。

```bash
pip install wandb[media]
```

## 画像

入力・出力画像、フィルターウェイトや活性値など、様々な画像情報を記録して比較できます。

{{< img src="/images/track/log_images.png" alt="オートエンコーダの入力・出力" >}}

画像は NumPy 配列・PIL 画像・ファイルパスから直接ログ可能です。

各ステップで画像をログすると、UI上でその画像を保存・表示します。画像パネルを展開し、ステップスライダーで異なるステップ間の画像を比較できます。モデルの出力がトレーニング中にどのように変化するかを簡単に追跡できます。

{{% alert %}}
トレーニング時のログや結果閲覧時にボトルネックとならないよう、1ステップにつき50枚未満の画像での記録を推奨します。
{{% /alert %}}

{{< tabpane text=true >}}
   {{% tab header="配列を画像として記録" %}}
NumPy 配列などを直接 wandb.Image に渡して画像化できます（例: [`torchvision`の`make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid) を利用）。

配列は [Pillow](https://pillow.readthedocs.io/en/stable/index.html) を使って png に変換されます。

```python
import wandb

with wandb.init(project="image-log-example") as run:

    images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

    run.log({"examples": images})
```

最後の次元が1ならグレースケール、3ならRGB、4ならRGBAと自動判定します。値がfloat型なら `0` から `255` へ変換されます。異なる正規化を行いたい場合は [`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) を指定するか、次の「PIL 画像での記録」タブにある手順で [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を直接渡してください。
   {{% /tab %}}
   {{% tab header="PIL 画像での記録" %}}
配列から画像への変換を細かく制御したい場合は、まず自分で [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を生成し、それを渡します。

```python
from PIL import Image

with wandb.init(project="") as run:
    # NumPy配列からPIL画像を生成
    image = Image.fromarray(image_array)

    # 必要に応じてRGBへ変換
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 画像をログ
    run.log({"example": wandb.Image(image, caption="My Image")})
```

   {{% /tab %}}
   {{% tab header="ファイルから画像を記録" %}}
さらに自由に制御したい場合、画像を任意の方法で生成し、ディスク保存後にファイルパス指定で記録できます。

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
   {{% tab header="セマンティックセグメンテーションマスク" %}}
セマンティックセグメンテーションのマスク（真値や予測ラベル）をUI上でオーバーレイ表示し、不透明度変更・時間による変化の確認などが可能です。

{{< img src="/images/track/semantic_segmentation.gif" alt="インタラクティブなマスク表示" >}}

マスクのオーバーレイを記録する場合は、`wandb.Image` の `masks` 引数に以下を含む辞書を渡します:

* マスク画像本体はいずれかのキーで指定
  * `"mask_data"`: 各ピクセルのクラスIDを持つ2次元NumPy配列
  * `"path"`: マスク画像ファイルのパス（文字列）
* `"class_labels"`: （オプション）ラベルIDから人間可読名への辞書

複数のマスクを記録したい場合は、複数のキーを持つマスク辞書をログします。コード例は以下の通りです。

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

各 key のセグメンテーションマスクは各ステップ（run.log() の呼び出し毎）で定義されます。
- 同じ key で異なる値がステップごとに記録された場合、画像には常に最新の値のみが適用されます。
- 異なる key のマスクが記録された場合、その step で定義済みの key のみがオーバーレイとして表示されます。他 step でのみ定義された key の ON/OFF は画像自体には影響しません。
   {{% /tab %}}
    {{% tab header="バウンディングボックス" %}}
画像と一緒にバウンディングボックスを記録し、UI でボックス種別ごとにフィルタ・トグル切り替えできます。

{{< img src="/images/track/bb-docs.jpeg" alt="バウンディングボックスの例" >}}

[ライブ例を見る](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログするには、`wandb.Image` の `boxes` 引数に次の形式の辞書を渡します:

* `box_data`: 各ボックスごとの辞書リスト。書式は以下。
  * `position`: ボックスの位置とサイズ。2つの書式が選択可で、混在可。
    * _方法1:_ `{"minX", "maxX", "minY", "maxY"}` 上下左右の境界座標を指定
    * _方法2:_ `{"middle", "width", "height"}` 「中心座標 `[x,y]` 」と幅高さ（スカラー値）
  * `class_id`: このボックスのクラス ID（整数）。下記`class_labels` と対応
  * `scores`: スコア名と値の辞書。UIでボックスをフィルタする際に使用可能
  * `domain`: ボックス座標の単位。**ピクセルで指定する場合は "pixel" と記入**
    デフォルトは画像内の割合（0〜1の小数）
  * `box_caption`: （オプション）ボックス上に表示される説明
* `class_labels`: （オプション）`class_id` からラベル文字列への辞書。指定が無い場合は`class_0`等が自動生成

サンプルコード:

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
                    # 比率(デフォルト)で表現されたボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセル空間（domain指定）のボックスも追加例
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 必要なだけボックスを追加
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 意味的なまとまりごとに一意の key でログ
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



## テーブルでの画像オーバーレイ

{{< tabpane text=true >}}
   {{% tab header="セマンティックセグメンテーションマスク" %}}
{{< img src="/images/track/Segmentation_Masks.gif" alt="テーブル内のセマンティックマスク例" >}}

テーブルにセグメンテーションマスクを記録する場合、各行に対して `wandb.Image` オブジェクトを準備します。

例は以下になります。

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
{{< img src="/images/track/Bounding_Boxes.gif" alt="テーブル内のバウンディングボックス例" >}}

テーブル内でバウンディングボックス付き画像を表示するには、各行ごとに `wandb.Image` オブジェクトを作成します。

下記が例です。

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
リスト・配列・テンソルなど数値の並びを第一引数に指定した場合、自動的に flatten し `np.histogram` でヒストグラム化します。`num_bins` キーワード引数でビン数（デフォルト 64・最大 512）も指定可能です。

UI では、横軸にトレーニングステップ、縦軸にメトリック値、色でカウントを表現するヒートマップ的プロットとなり、トレーニング過程を比較しやすくなります。単発ヒストグラムの記録については本パネルの「Summaryでのヒストグラム」タブを参照してください。

```python
run.log({"gradients": wandb.Histogram(grads)})
```

{{< img src="/images/track/histograms.png" alt="GAN判別子の勾配ヒストグラム" >}}   
   {{% /tab %}}
   {{% tab header="柔軟なヒストグラム記録" %}}
ヒストグラム化を任意で細かく制御したい場合には、自分で `np.histogram` を呼び出し、その戻り値タプルを `np_histogram` キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
run.log({"gradients": wandb.Histogram(np_hist_grads)})
```
   {{% /tab %}}
{{< /tabpane >}}



ヒストグラムを summary に記録した場合、[Run Page]({{< relref "/guides/models/track/runs/" >}}) の Overview タブに表示されます。history に記録された場合は Charts タブで時間推移付きのヒートマップ表示となります。

## 3D 可視化

3D 点群や Lidar シーン（バウンディングボックス付き）を記録できます。描画したい点群を NumPy 配列で渡します。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

run.log({"point_cloud": wandb.Object3D(point_cloud)})
```

{{% alert %}}
W&B UI は 30 万点を超えるポイントデータをカットオフします。
{{% /alert %}}

#### NumPy 配列フォーマット

色やカテゴリ表現に応じて3通りの配列形式に対応しています

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c: 1〜14 のカテゴリ`（セグメンテーションに便利です）
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b`: 色成分 0-255 の整数値（RGB）

#### Pythonオブジェクト

下記スキーマ例のように、Pythonオブジェクトで定義して [ `from_point_cloud` メソッド]({{< relref "/ref/python/sdk/data-types/Object3D/#from_point_cloud" >}}) に渡すことも可能です。

* `points`：上記と同様の NumPy 配列
* `boxes`：Pythonの辞書で8点分の corner 座標、ラベル、ボックス色等を記述したリスト
  * `corners` - 8つの corner座標リスト
  * `label` - ボックスに表示される文字列（オプション）
  * `color` - ボックス色 (RGB)
  * `score` - フィルタ用のスコア（例えばスコア > 0.75のみ表示など。オプション）
* `type`：描画シーンタイプ(現在は "lidar/beta" のみ対応)

```python
point_list = [
    [
        2566.571924017235, # x
        746.7817289698219, # y
        -15.269245470863748,# z
        76.5, # 赤
        127.5, # 緑
        89.46617199365393 # 青
    ],
    [ 2566.592983606823, 746.6791987335685, -15.275803826279521, 76.5, 127.5, 89.45471117247024 ],
    [ 2566.616361739416, 746.4903185513501, -15.28628929674075, 76.5, 127.5, 89.41336375503832 ],
    [ 2561.706014951675, 744.5349468458361, -14.877496818222781, 76.5, 127.5, 82.21868245418283 ],
    [ 2561.5281847916694, 744.2546118233013, -14.867862032341005, 76.5, 127.5, 81.87824684536432 ],
    [ 2561.3693562897465, 744.1804761656741, -14.854129178142523, 76.5, 127.5, 81.64137897587152 ],
    [ 2561.6093071504515, 744.0287526628543, -14.882135189841177, 76.5, 127.5, 81.89871499537098 ],
    # ... 以降省略
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
         "color": [0, 0, 255], # ボックスの色 (RGB)
         "label": "car", # 表示ラベル
         "score": 0.6 # ボックスの数値スコア
     }],
     vectors = [
        {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5], "color": [255, 0, 0]}, # 色を省略可
     ],
     point_cloud_type = "lidar/beta",
)})
```

点群表示時はCtrlキー+マウスで空間内を移動できます。

#### ポイントクラウドファイル

ポイントクラウドデータ(JSON)を [ `from_file` メソッド]({{< relref "/ref/python/sdk/data-types/Object3D/#from_file" >}}) で読み込むこともできます。

```python
run.log({"my_cloud_from_file": wandb.Object3D.from_file(
     "./my_point_cloud.pts.json"
)})
```

ファイルの例:

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

[配列フォーマットの説明]({{< relref "#numpy-array-formats" >}}) の通り、 [ `from_numpy` メソッド]({{< relref "/ref/python/sdk/data-types/Object3D/#from_numpy" >}}) に配列を渡して点群を定義することもできます。

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

分子データ（`pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, `mmtf` の10種類）もログできます。

また SMILES 文字列や [`rdkit`](https://www.rdkit.org/docs/index.html) の`mol`ファイルや `rdkit.Chem.rdchem.Mol` オブジェクトにも対応しています。

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

run が完了すると、UI上で分子の3D可視化・インタラクションが行えます。

[AlphaFold を用いたライブ例を見る](https://wandb.me/alphafold-workspace)

{{< img src="/images/track/docs-molecule.png" alt="分子構造例" >}}
  </TabItem>
</Tabs>

### PNG画像

[`wandb.Image`]({{< relref "/ref/python/sdk/data-types/image.md" >}}) は `numpy` 配列や `PILImage` インスタンスをデフォルトで PNG へ変換します。

```python
run.log({"example": wandb.Image(...)})
# 複数画像の場合
run.log({"example": [wandb.Image(...) for img in images]})
```

### 動画

動画は [`wandb.Video`]({{< relref "/ref/python/sdk/data-types/Video" >}}) でログします。

```python
run.log({"example": wandb.Video("myvideo.mp4")})
```

記録した動画は、プロジェクトワークスペースや run ページ、または report のメディアブラウザで閲覧できます。追加表示する場合は **Add visualization** をクリックしてください。

## 分子の2Dビュー

分子の2D画像を [`wandb.Image`]({{< relref "/ref/python/sdk/data-types/image.md" >}}) と [`rdkit`](https://www.rdkit.org/docs/index.html) を組み合わせてログできます。

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

run.log({"acetic_acid": wandb.Image(pil_image)})
```


## その他メディア

W&B ではその他さまざまなメディアタイプの記録もサポートしています。

### 音声

```python
run.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

1ステップあたり最大100個の音声クリップが記録可能です。さらに詳しい情報は、[`audio-file`]({{< relref "/ref/query-panel/audio-file.md" >}}) を参照ください。

### 動画

```python
run.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

NumPy配列の場合は `time, channels, width, height` の順で扱い、デフォルトでは4fpsのgifを生成します（numpy利用時は [`ffmpeg`](https://www.ffmpeg.org) および [`moviepy`](https://pypi.org/project/moviepy/) が必要）。サポートフォーマット: `"gif"`, `"mp4"`, `"webm"`, `"ogg"`。文字列でパス指定の際は該当拡張子が wandb 対応か検証されますし、BytesIO オブジェクト指定時は指定拡張子のテンポラリファイルが作られます。

動画は W&B [Run]({{< relref "/guides/models/track/runs/" >}}) および [Project]({{< relref "/guides/models/track/project-page.md" >}}) ページの Media セクションに表示されます。

より詳しい情報は[`video-file`]({{< relref "/ref/query-panel/video-file" >}}) をご覧ください。

### テキスト

UIでテーブルにテキストを表示するには `wandb.Table` を活用します。デフォルトのカラムは `["Input", "Output", "Expected"]`。パフォーマンス維持のためデフォルト最大行数は1万件ですが、`wandb.Table.MAX_ROWS = {DESIRED_MAX}` で明示的に増減できます。

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

pandas の DataFrame オブジェクトも渡せます。

```python
table = wandb.Table(dataframe=my_dataframe)
```

さらに詳しい情報は[`string`]({{< relref "/ref/query-panel/" >}}) を参照ください。

### HTML

```python
run.log({"custom_file": wandb.Html(open("some.html"))})
run.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

カスタムHTMLを任意の key でログ可能です。Run ページに専用 HTML パネルができます。デフォルトでスタイルも自動挿入されますが、`inject=False` で無効化できます。

```python
run.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

さらに詳しい情報は[`html-file`]({{< relref "/ref/query-panel/html-file" >}}) をご覧ください。