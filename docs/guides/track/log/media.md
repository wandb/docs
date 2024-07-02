---
description: 3Dポイントクラウドや分子からHTMLやヒストグラムまで、リッチメディアをログします
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Log Media & Objects

画像、ビデオ、音声など、さまざまなメディアをサポートしています。豊富なメディアをログして結果を詳細に検討し、Runs、Models、Datasetsを視覚的に比較しましょう。具体例やガイドについては以下をお読みください。

:::info
メディアタイプのリファレンスドキュメントをお探しですか？[このページ](../../../ref/python/data-types/README.md)をご参照ください。
:::



:::info
これらのメディアオブジェクトをログするための動作するコードは、[このColab Notebook](http://wandb.me/media-colab)で確認できます。また、結果がどう見えるかを[こちら](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)で確認できます。さらに、ビデオチュートリアルもリンクされています。
:::

## Images

画像をログして、入力、出力、フィルタの重み、活性化などを追跡しましょう！

![オートエンコーダーネットワークのインペインティングの入力と出力](/images/track/log_images.png)

画像はNumPy配列、PIL画像、またはファイルシステムから直接ログできます。

:::info
トレーニング中にログがボトルネックとならないように、一度のステップでログする画像は50枚未満にすることをお勧めします。また、結果を表示する際にも画像の読み込みがボトルネックにならないようにします。
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: 'Logging Arrays as Images', value: 'arrays'},
    {label: 'Logging PIL Images', value: 'pil_images'},
    {label: 'Logging Images from Files', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

例えば [`torchvision` の `make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid) を使って手動で画像を作成する場合、配列を直接提供します。

配列は [Pillow](https://pillow.readthedocs.io/en/stable/index.html) を使ってpngに変換されます。

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

画像がグレースケールである場合は最後の次元が1、RGBの場合は3、RGBAの場合は4であると仮定します。配列が浮動小数点数の場合、これを `0` から `255` の範囲の整数に変換します。異なる方法で画像を正規化したい場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) を手動で指定するか、"Logging PIL Images" タブで説明されているように、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を直接提供できます。
  </TabItem>
  <TabItem value="pil_images">

配列から画像への変換を完全にコントロールするために、自分で [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を構築し、直接提供します。

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
さらにコントロールが必要な場合は、任意の方法で画像を作成し、ディスクに保存してファイルパスを提供します。

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## Image Overlays

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

セマンティックセグメンテーションマスクをログし、W&BのUIでそれらとインタラクトできます（不透明度の変更、時間経過の表示など）。

![W&B UIでのインタラクティブなマスク表示](/images/track/semantic_segmentation.gif)

オーバーレイをログするには、以下のキーと値を含む辞書を `wandb.Image` の `masks` キーワード引数に提供する必要があります。

* 画像マスクを表す2つのキーのいずれか:
  * `"mask_data"`: 各ピクセルのクラスラベルを含む2次元NumPy配列
  * `"path"`: 保存された画像マスクファイルへのパス（文字列）
* `"class_labels"`: （オプション）画像マスク内のクラスラベルとその読みやすいクラス名をマッピングする辞書

複数のマスクをログするには、複数のキーを持つマスク辞書をログします。以下のコードスニペットを参照してください。

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
画像にバウンディングボックスをログし、UIでフィルタやトグルを使用して表示します。

![](@site/static/images/track/bb-docs.jpeg)

[ライブ例を見る →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログするには、以下のキーと値を含む辞書を `wandb.Image` の `boxes` キーワード引数に提供する必要があります。

* `box_data`: 各ボックスの辞書のリスト。ボックス辞書のフォーマットは以下に示します。
  * `position`: ボックスの位置とサイズを表す辞書。以下の2つの形式のいずれかを用います。ボックスは同じ形式を使用する必要はありません。
    * オプション1: `{"minX", "maxX", "minY", "maxY"}`. 各ボックスの次元の上限と下限の座標を定義します。
    * オプション2: `{"middle", "width", "height"}`. 中心座標を `[x, y]` として指定し、`width` および `height` をスカラーとして指定します。
  * `class_id`: ボックスのクラス識別子を表す整数。以下の `class_labels` キーを参照。
  * `scores`: スコアのラベルと数値の辞書。UIでのフィルタリングに使用できます。
  * `domain`: ボックス座標の単位/形式を指定します。座標がピクセル空間（つまり画像の次元内の整数）で表現されている場合、これを "pixel" に設定します。デフォルトでは、ボックス座標は画像の比率/割合（0から1の浮動小数点数）として表現されているとみなされます。
  * `box_caption`: （オプション）このボックスのラベルとして表示される文字列
* `class_labels`: （オプション）`class_id` を文字列にマッピングする辞書。デフォルトでは `class_0`、`class_1` などのクラスラベルが生成されます。

次の例を確認してください：

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
                    # デフォルトの相対的な領域で表現されたボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセル領域で表現された別のボックス
                    # （説明のために示したものであり、通常は
                    # すべてのボックスが同じ領域/形式で表現されます）
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # 必要に応じてボックスをログ
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 意味のあるボックスのグループごとにユニークなキー名でログ
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## Image Overlays in Tables

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![Tablesでのインタラクティブなセグメンテーションマスク](/images/track/Segmentation_Masks.gif)

Tables にセグメンテーションマスクをログするには、各行に `wandb.Image` オブジェクトを提供する必要があります。

以下のコードスニペットに例を示します：

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


![Tablesでのインタラクティブなバウンディングボックス](/images/track/Bounding_Boxes.gif)

Tables にバウンディングボックス付きの画像をログするには、各行に `wandb.Image` オブジェクトを提供する必要があります。

以下のコードスニペットに例を示します：

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

## Histograms

<Tabs
  defaultValue="histogram_logging"
  values={[
    {label: 'Basic Histogram Logging', value: 'histogram_logging'},
    {label: 'Flexible Histogram Logging', value: 'flexible_histogram'},
    {label: 'Histograms in Summary', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">
  
数値のシーケンス（例：リスト、配列、テンソル）が最初の引数として提供された場合、自動的に `np.histogram` を呼び出してヒストグラムを構築します。すべての配列/テンソルはフラット化されます。デフォルトの64ビンを上書きするためにオプションの `num_bins` キーワード引数を使用できます。サポートされている最大ビン数は512です。

UIでは、ヒストグラムはトレーニングステップをx軸、メトリック値をy軸にプロットし、ログされたヒストグラムを比較しやすくするために色でカウントを表現します。詳細については、このパネルの「Histograms in Summary」タブを参照してください。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GANの判別器の勾配](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

さらに詳細なコントロールが必要な場合は、`np.histogram` を呼び出し、返されたタプルを `np_histogram` キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # サマリーにのみ表示される場合、overviewタブでのみ表示
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

サマリーにあるヒストグラムは[Run Page](../../app/pages/run-page.md)のoverviewタブに表示されます。ヒストリーにある場合は、Chartsタブで時間経過にわたるビンのヒートマップを表示します。

## 3D Visualizations

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D Object', value: '3d_object'},
    {label: 'Point Clouds', value: 'point_clouds'},
    {label: 'Molecules', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

`'obj'`, `'gltf'`, `'glb'`, `'babylon'`, `'stl'`, `'pts.json'` 形式のファイルをログすると、runが終了したときにUIで表示されます。

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

![ヘッドフォンポイントクラウドの正解と予測](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[ライブ例を見る →](https://app.wandb.ai/nbaryd/SparseConvNet-examples\_3d\_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

3DポイントクラウドとLidarシーンをバウンディングボックス付きでログします。レンダリング用の座標と色を含むNumPy配列を渡します。UIでは、ポイントを300,000点に制限します。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

柔軟なカラースキームのために、3つの異なる形状のNumPy配列がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| cはカテゴリ` の範囲 `[1, 14]` （セグメンテーションに便利）
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は赤、緑、青の各色チャンネルの範囲 `[0,255]`

以下はログコードの例です：

* `points`は単純なポイントクラウドレンダラで示される形式のNumPy配列です。
* `boxes`は3つの属性を持つPython辞書のNumPy配列です：
  * `corners`- 8つのコーナーリスト
  * `label`- ボックス上に表示するラベルを表す文字列（オプション）
  * `color`- ボックスの色を表すrgb値
* `type`はシーンタイプを表す文字列です。現在サポートされている唯一の値は `lidar/beta` です。

```python
# W&Bにポイントとボックスをログ
point_scene = wandb.Object3D(
    {
        "type": "lidar/beta",
        "points": np.array(  # ポイントを追加、ポイントクラウドの如く
            [[0.4, 1, 1.3], [1, 1, 1], [1.2, 1, 1.2]]
        ),
        "boxes": np.array(  # 3Dボックスを描画
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
        "vectors": np.array(  # 3Dベクトルを追加
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

10種類のファイルタイプで分子データをログできます：`pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, または `mmtf.`

W&Bはまた、SMILES文字列、[`rdkit`](https://www.rdkit.org/docs/index.html) `mol` ファイル、および `rdkit.Chem.rdchem.Mol` オブジェクトからの分子データのログをサポートしています。

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

runが終了すると、UIで分子の3D可視化とインタラクトできるようになります。

[AlphaFoldを使用したライブ例を見る →](http://wandb.me/alphafold-workspace)

![分子の2Dビュー](@site/static/images/track/docs-molecule.png)

  </TabItem>
</Tabs>

## Other Media

W&Bはその他のさまざまなメディアタイプのログもサポートしています。

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

ログできる音声クリップの最大数はステップあたり100です。

  </TabItem>
  <TabItem value="video">

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

NumPy配列が提供された場合、次元は順に時間、チャンネル、幅、高さであると仮定します。デフォルトでは4fpsのgif画像を作成します（NumPyオブジェクトを渡す場合、[`ffmpeg`](https://www.ffmpeg.org) および `moviepy`](https://pypi.org/project/moviepy/) Pythonライブラリが必要です）。サポートされている形式は `"gif"`, `"mp4"`, `"webm"`, および `"ogg"` です。`wandb.Video` に文字列を渡すと、そのファイルが存在し、サポートされている形式であることをチェックしてから、wandbにアップロードします。`BytesIO` オブジェクトを渡すと、指定された形式の拡張子で一時ファイルが作成されます。

W&Bの[Run](../../app/pages/run-page.md) および[Project](../../app/pages/project-page.md) ページで、ビデオをメディアセクションで確認できます。

  </TabItem>
  <TabItem value="text">

`wandb.Table` を使用してテキストをTablesにログして、UIに表示します。デフォルトでは、カラムヘッダーは `["Input", "Output", "Expected"]` です。最適なUIパフォーマンスを確保するために、デフォルトの最大行数は10,000に設定されていますが、`wandb.Table.MAX_ROWS = {DESIRED_MAX}` で最大値を明示的に上書きできます。

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# 方法1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# 方法2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

pandas `DataFrame` オブジェクトも渡すことができます。

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

カスタムHTMLを任意のキーにログでき、これによりrunページにHTMLパネルが表示されます。デフォルトでスタイルが注入されますが、`inject=False` を渡すことでデフォルトスタイルを無効にできます。

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## Frequently Asked Questions

### 画像やメディアをエポックやステップごとに比較するにはどうすればよいですか？

ステップごとに画像をログするたびに、それを保存してUIに表示します。画像パネルを展開し、ステップスライダーを使用して異なるステップの画像を確認できます。これにより、トレーニング中にモデルの出力がどのように変化するかを簡単に比較できます。

### プロジェクトにW&Bを統合したいが、画像やメディアをアップロードしたくない場合はどうすればよいですか？

W&Bはスカラーのみをログするプロジェクトにも使用できます。アップロードしたいファイルやデータは明示的に指定します。画像をログしない[PyTorchの簡単な例](http://wandb.me/pytorch-colab)をご覧ください。

### PNGをどうやってログしますか？

[`wandb.Image`](../../../ref/python/data-types/image.md)は、`numpy` 配列や `PILImage` のインスタンスをデフォルトでPNGに変換します。

```python
wandb.log({"example": wandb.Image(...)})
# または複数の画像をログ
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### ビデオをどうやってログしますか？

ビデオは [`wandb.Video`](../../../ref/python/data-types/video.md) データタイプを使用してログされます：

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

これでメディアブラウザでビデオを見ることができます。プロジェクトworkspace、run workspace、またはレポートに移動し、「Add visualization」をクリックして豊富なメディアパネルを追加します。

### ポイントクラウド内でナビゲートおよびズームインするにはどうすればよいですか？

コントロールキーを押しながらマウスを使用してスペース内を移動します。

### 分子の2Dビューをどうやってログしますか？

[`wandb.Image`](../../../ref/python/data-types/image.md) データタイプと [`rdkit`](https://www.rdkit.org/docs/index.html) を使用して分子の2Dビューをログできます：

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

