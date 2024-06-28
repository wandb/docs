---
description: リッチメディアをログする：3D点群や分子からHTMLやヒストグラムまで
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Log Media & Objects

画像、ビデオ、オーディオなどのメディアに対応しています。リッチメディアをログして、結果を探索し、run、モデル、データセットを視覚的に比較しましょう。詳細な例とハウツーガイドについては以下をお読みください。

:::info
メディアタイプのリファレンスドキュメントをお探しですか？[こちらのページ](../../../ref/python/data-types/README.md)をご覧ください。
:::

:::info
これらのメディアオブジェクトをログするための動作中のコードは、この [Colab Notebook](http://wandb.me/media-colab) で確認でき、結果は wandb.ai で [こちら](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA) で確認できます。また、上記のリンクからビデオチュートリアルも参照できます。
:::

## 画像

入力、出力、フィルタの重み、活性化などを追跡するために画像をログしましょう。

![オートエンコーダーネットワークによるインペインティングの入力と出力。](/images/track/log_images.png)

画像はNumPy配列から、PIL画像として、またはファイルシステムから直接ログすることができます。

:::info
トレーニング中にログがボトルネックにならないように、1ステップあたり50枚未満の画像をログすることを推奨します。また、結果表示時に画像読み込みがボトルネックになるのを防ぎます。
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: '配列を画像としてログ', value: 'arrays'},
    {label: 'PIL画像をログ', value: 'pil_images'},
    {label: 'ファイルから画像をログ', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

[`torchvision` からの `make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid) を使用して手動で画像を作成するときに、配列を直接提供します。

配列は [Pillow](https://pillow.readthedocs.io/en/stable/index.html) を使用してpngに変換されます。

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

最後の次元が1の場合はグレースケール画像、3の場合はRGB画像、4の場合はRGBA画像と見なします。配列が浮動小数点を含む場合は、`0`から`255`の間の整数に変換します。画像を異なる方法で正規化したい場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) を手動で指定するか、「PIL画像をログ」タブで示されているように単に [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を提供することができます。
  </TabItem>
  <TabItem value="pil_images">

配列を画像に変換するための完全な制御が必要な場合、自分で [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を構築し、それを直接提供します。

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
さらに詳細な制御が必要な場合、好きな方法で画像を作成し、ディスクに保存してファイルパスを提供します。

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## 画像のオーバーレイ

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'セマンティックセグメンテーションマスク', value: 'segmentation_masks'},
    {label: 'バウンディングボックス', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

セマンティックセグメンテーションマスクをログし、W&BのUIを通じて操作（不透明度の変更、時間経過による変化の表示など）することができます。

![W&B UIでのインタラクティブなマスク表示。](/images/track/semantic_segmentation.gif)

オーバーレイをログするには、以下のキーと値を含む辞書を `wandb.Image` の `masks` キーワード引数に提供する必要があります。

* 画像マスクを表す2つのキーのうちの1つ：
  * `"mask_data"`: 各ピクセルの整数クラスラベルを含む2D NumPy配列
  * `"path"`: 保存された画像マスクファイルへのパス（文字列）
* `"class_labels"`: （オプション）画像マスク内の整数クラスラベルを読みやすいクラス名にマッピングする辞書

複数のマスクをログするには、以下のコードスニペットのように、複数のキーを持つマスク辞書をログします。

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
画像にバウンディングボックスをログし、UIでフィルタと切り替えを使用して異なるボックスセットを動的に視覚化します。

![](@site/static/images/track/bb-docs.jpeg)

[ライブ例を見る →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログするには、以下のキーと値を含む辞書を `wandb.Image` の `boxes` キーワード引数に提供する必要があります：

* `box_data`: 各ボックスに対する辞書のリスト。ボックス辞書の形式は以下で説明されています。
  * `position`: 位置とサイズを表す辞書。以下の2つの形式のどちらかで、ボックスごとに異なる形式を使用する必要はありません。
    * _オプション1:_ `{"minX", "maxX", "minY", "maxY"}`。各次元の上下限を定義する座標を提供します。
    * _オプション2:_ `{"middle", "width", "height"}`。`middle` 座標を `[x,y]` として、`width` と `height` をスカラーとして提供します。
  * `class_id`: ボックスのクラス識別を表す整数。以下の `class_labels` キーを参照してください。
  * `scores`: スコアの文字列ラベルと数値を含む辞書。UIでボックスをフィルタリングするために使用できます。
  * `domain`: ボックス座標の単位/形式を指定します。ボックス座標がピクセル空間（すなわち画像次元内の整数）で表されている場合は、**これを "pixel" に設定してください**。デフォルトでは、ドメインは画像の分数/割合（0と1の間の浮動小数点数）であると見なされます。
  * `box_caption`: (オプション) このボックスに表示されるラベルテキストとしての文字列
* `class_labels`: (オプション) `class_id` を文字列にマッピングする辞書。デフォルトでは、`class_0`、`class_1` などのクラスラベルが生成されます。

この例を参考にしてください：

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
                    # ピクセルのドメインで表現された別のボックス
                    # （例示のみで、すべてのボックスは同じドメイン/形式である可能性が高い）
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 必要に応じて多くのボックスログ
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 意味のある各ボックスグループをユニークなキー名でログ
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
    {label: 'セマンティックセグメンテーションマスク', value: 'segmentation_masks'},
    {label: 'バウンディングボックス', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![テーブル内のインタラクティブなセマンティックセグメンテーションマスク](/images/track/Segmentation_Masks.gif)

テーブル内にセマンティックセグメンテーションマスクをログするには、テーブルの各行に対して `wandb.Image` オブジェクトを提供する必要があります。

コードスニペットに例があります：

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

テーブル内にバウンディングボックス付きの画像をログするには、テーブルの各行に対して `wandb.Image` オブジェクトを提供する必要があります。

コードスニペットに例があります：

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
    {label: '基本的なヒストグラムログ', value: 'histogram_logging'},
    {label: '柔軟なヒストグラムログ', value: 'flexible_histogram'},
    {label: '要約におけるヒストグラム', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">
  
数値のシーケンス（例：リスト、配列、テンソル）が最初の引数として提供されると、自動的に `np.histogram` を呼び出してヒストグラムを構築します。すべての配列/テンソルはフラット化されることに注意してください。オプションの `num_bins` キーワード引数を使用してデフォルトの`64`ビンを上書きできます。サポートされている最大のビン数は`512`です。

UIでは、トレーニングのステップをx軸、メトリックの値をy軸、カウントを色で表してヒストグラムをプロットし、トレーニング全体でログされたヒストグラムの比較を容易にします。「要約におけるヒストグラム」タブでは、一度だけログされるヒストグラムの詳細について説明しています。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GANの識別器の勾配。](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

さらに制御が必要な場合、`np.histogram` を呼び出して返されたタプルを `np_histogram` キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # 要約にのみ表示される場合、overviewタブのみに表示
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

ヒストグラムが要約に含まれている場合は、[Run Page](../../app/pages/run-page.md) の Overview tab に表示されます。履歴に含まれている場合は、Charts tab に時間経過ごとのビンのヒートマップをプロットします。

## 3D Visualizations

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D Object', value: '3d_object'},
    {label: 'Point Clouds', value: 'point_clouds'},
    {label: 'Molecules', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

ファイル形式 `'obj', 'gltf', 'glb', 'babylon', 'stl', 'pts.json'` のファイルをログに記録し、run が終了するときに UI に表示します。

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

![ヘッドフォンのポイントクラウドの正解と予測](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[ライブ例を見る →](https://app.wandb.ai/nbaryd/SparseConvNet-examples\_3d\_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

バウンディングボックスを伴う3DポイントクラウドとLidarシーンをログに記録します。レンダリングするポイントの座標と色を含む NumPy 配列を渡します。UI では、ポイント数を 300,000 に制限します。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

柔軟なカラースキームのために、3つの異なる形状の NumPy 配列がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c はカテゴリ` `[1, 14]` の範囲 (セグメンテーションに有用)
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は赤, 緑, 青の色チャンネル `[0,255]` の範囲の値。

以下はログを記録するコードの例です：

* `points`は上記のシンプルなポイントクラウドレンダラーと同じ形式の NumPy 配列です。
* `boxes` は3つの属性を持つ Python 辞書の NumPy 配列です：
  * `corners`- 8つのコーナーのリスト
  * `label`- ボックスにレンダリングされるラベルを表す文字列 (オプション)
  * `color`- ボックスの色を表す RGB 値
* `type` はレンダリングするシーンタイプを表す文字列です。現在サポートされている値は `lidar/beta` のみです。

```python
# W&B にポイントとボックスをログする
point_scene = wandb.Object3D(
    {
        "type": "lidar/beta",
        "points": np.array(  # ポイントを追加、ポイントクラウドのように
            [[0.4, 1, 1.3], [1, 1, 1], [1.2, 1, 1.2]]
        ),
        "boxes": np.array(  # 3D ボックスを描画
            [
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ],
                    "label": "Box",
                    "color": [123, 321, 111],
                },
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 2, 0],
                        [0, 0, 2],
                        [2, 0, 0],
                        [2, 2, 0],
                        [0, 2, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    ],
                    "label": "Box-2",
                    "color": [111, 321, 0],
                },
            ]
        ),
        "vectors": np.array(  # 3D ベクトルを追加
            [{"start": [0, 0, 0], "end": [0.1, 0.2, 0.5]}]
        ),
    }
)
wandb.log({"point_scene": point_scene})
```
  </TabItem>
  <TabItem value="molecules">

```python
wandb.log({"protein": wandb.Molecule("6lu7.pdb")})
```

10種類のファイルタイプ `pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, または `mmtf` 形式で分子データをログします。

W&B はまた、SMILES 文字列、[`rdkit`](https://www.rdkit.org/docs/index.html) の `mol` ファイル、および `rdkit.Chem.rdchem.Mol` オブジェクトから分子データをログすることもサポートします。

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

run が終了すると、UI で分子の3D可視化と対話できるようになります。

[AlphaFoldを使用したライブ例を見る →](http://wandb.me/alphafold-workspace)

![](@site/static/images/track/docs-molecule.png)
  </TabItem>
</Tabs>

## その他のメディア

W&B はさまざまな他のメディアタイプのログをサポートします。

<Tabs
  defaultValue="audio"
  values={[
    {label: 'Audio', value: 'audio'},
    {label: 'Video', value: 'video'},
    {label: 'Text', value: 'text'},
    {label: 'HTML', value: 'html'},
  ]}>
  <TabItem value="audio">

```python
wandb.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

ステップごとにログに記録できるオーディオクリップの最大数は100です。

  </TabItem>
  <TabItem value="video">

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy 配列が提供される場合、次の順序で次元が設定されていると仮定します：時間、チャンネル、幅、高さ。デフォルトでは、4 fps の gif 画像を作成します（numpy オブジェクトを渡す場合には [`ffmpeg`](https://www.ffmpeg.org) と [`moviepy`](https://pypi.org/project/moviepy/) Python ライブラリが必要）。サポートされている形式は `"gif"`, `"mp4"`, `"webm"`, `"ogg"` です。`wandb.Video` に文字列を渡すと、ファイルが存在しておりサポートされている形式であることを確認してから wandb にアップロードします。`BytesIO` オブジェクトを渡すと、指定された形式を拡張子とする一時ファイルが作成されます。

W&B の [Run](../../app/pages/run-page.md) および [Project](../../app/pages/project-page.md) ページでは、メディアセクションでビデオが表示されます。

  </TabItem>
  <TabItem value="text">

`wandb.Table` を使用して、UI に表示するためのテキストをテーブルにログします。デフォルトでは、列ヘッダーは `["Input", "Output", "Expected"]` です。最適な UI パフォーマンスを確保するため、デフォルトの最大行数は 10,000 に設定されています。ただし、ユーザーは `wandb.Table.MAX_ROWS = {DESIRED_MAX}` を使って最大値を明示的にオーバーライドできます。

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# 方法 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# 方法 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

pandas の `DataFrame` オブジェクトを渡すこともできます。

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

任意のキーにカスタム HTML をログでき、run ページに HTML パネルが表示されます。デフォルトでスタイルを注入しますが、`inject=False` を渡すことでデフォルトのスタイルを無効にできます。

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## よくある質問

### イメージやメディアをエポックやステップを超えてどうやって比較できますか？

ステップごとにイメージをログするたびに、それらを保存し、UI に表示します。イメージパネルを拡張し、ステップスライダーを使って異なるステップのイメージを確認してください。これにより、モデルの出力がトレーニング中にどのように変化するかを簡単に比較できます。

### W&B をプロジェクトに統合したいですが、イメージやメディアをアップロードしたくない場合はどうすればよいですか？

W&B はスカラーのみをログするプロジェクトでも使用できます。アップロードするファイルやデータを明示的に指定します。イメージをログしない [PyTorch のクイック例](http://wandb.me/pytorch-colab) をご覧ください。

### PNG をどうやってログしますか？

[`wandb.Image`](../../../ref/python/data-types/image.md) は `numpy` 配列または `PILImage` インスタンスをデフォルトで PNG に変換します。

```python
wandb.log({"example": wandb.Image(...)})
# または複数のイメージ
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### ビデオをどうやってログしますか？

ビデオは [`wandb.Video`](../../../ref/python/data-types/video.md) データ型を使用してログします：

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

これでメディアブラウザでビデオを閲覧できるようになります。プロジェクトワークスペース、runワークスペース、またはレポートに移動し、「可視化を追加」をクリックしてリッチメディアパネルを追加します。

### ポイントクラウドのナビゲートとズームインはどうやって行いますか？

コントロールキーを押しながらマウスを使ってスペース内を移動できます。

### 分子の2Dビューをどうやってログしますか？

[`wandb.Image`](../../../ref/python/data-types/image.md) データ型と [`rdkit`](https://www.rdkit.org/docs/index.html) を使用して分子の2Dビューをログできます：

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```