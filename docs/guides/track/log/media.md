---
description: 3Dポイントクラウドや分子からHTMLやヒストグラムまで、リッチメディアをログする
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Log Media & Objects

私たちは、画像、ビデオ、オーディオなどをサポートしています。リッチメディアをログに記録して結果を探り、runs、models、およびdatasetsを視覚的に比較しましょう。例とハウツーガイドについては以下をお読みください。

:::info
メディアタイプのリファレンスドキュメントをお探しですか？[このページ](../../../ref/python/data-types/README.md)をご覧ください。
:::

:::info
これらのメディアオブジェクトをログに記録するための動作コードは[このColab Notebook](http://wandb.me/media-colab)で確認できます。また、wandb.aiでの結果は[こちら](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)で確認でき、ビデオチュートリアルも上記リンクから参照できます。
:::

## 画像

画像をログに記録して入力、出力、フィルターの重み、アクティベーションなどをトラッキングしましょう！

![インペイント処理を行うオートエンコーダーネットワークの入力と出力。](/images/track/log_images.png)

画像はNumPy配列から直接、PIL画像として、またはファイルシステムからログに記録できます。

:::info
トレーニング中のログ記録がボトルネックにならないよう、1ステップあたり50枚未満の画像をログに記録することをお勧めします。また、結果を表示するときに画像の読み込みがボトルネックになるのを防ぎます。
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: 'Logging Arrays as Images', value: 'arrays'},
    {label: 'Logging PIL Images', value: 'pil_images'},
    {label: 'Logging Images from Files', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

`[`make_grid` from `torchvision`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)`などを利用して手動で画像を作成する際に直接配列を提供します。

配列は[Pillow](https://pillow.readthedocs.io/en/stable/index.html)を使用してpngに変換されます。

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

最後の次元が1の場合はグレースケール、3の場合はRGB、4の場合はRGBAとして画像を扱います。配列に浮動小数点が含まれている場合は、`0` から `255` までの整数に変換します。異なる方法で画像を正規化したい場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)を手動で指定するか、単に[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)を指定します。詳細はこのパネルの「Logging PIL Images」タブを参照してください。
  </TabItem>
  <TabItem value="pil_images">

配列から画像への変換を完全にコントロールするために、自身で[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)オブジェクトを構築し、それを直接提供します。

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
さらなるコントロールが必要な場合は、任意の方法で画像を作成し、ディスクに保存してファイルパスを提供します。

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## 画像オーバーレイ

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

セマンティックセグメンテーションマスクをログに記録し、W&B UIを介してこれらと対話（不透明度の変更、時間経過による変化の表示など）します。

![W&B UIでのインタラクティブなマスク表示](/images/track/semantic_segmentation.gif)

オーバーレイをログに記録するには、以下のキーと値を含む辞書を`wandb.Image`の`masks`キーワード引数に提供する必要があります。

* 画像マスクを表す2つのキーのうちの1つ:
  * `"mask_data"`: 各ピクセルの整数クラスラベルを含む2D NumPy配列
  * `"path"`: （文字列）保存された画像マスクファイルへのパス
* `"class_labels"`: （オプション）画像マスク内の整数クラスラベルを読み取り可能なクラス名にマッピングする辞書

複数のマスクをログに記録するには、以下のコードスニペットのように複数のキーを持つマスク辞書をログに記録します。

[ライブの例を参照 →](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[サンプルコード →](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix_P4J)

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

画像と一緒にバウンディングボックスをログに記録し、フィルタやトグルを使用してUIで動的に異なるボックスセットを可視化します。

![](@site/static/images/track/bb-docs.jpeg)

[ライブの例を参照 →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログに記録するには、以下のキーと値を含む辞書を`wandb.Image`の`boxes`キーワード引数に提供する必要があります。

* `box_data`: 各ボックスの辞書のリスト。ボックス辞書のフォーマットは以下で説明します。
  * `position`: ボックスの位置とサイズを表す辞書。2つのフォーマットのいずれかで記述します。すべてのボックスが同じフォーマットを使用する必要はありません。
    * _オプション1:_ `{"minX", "maxX", "minY", "maxY"}`。各次元の上限と下限を定義する座標のセットを提供します。
    * _オプション2:_ `{"middle", "width", "height"}`。`middle`座標を`[x,y]`として、`width`と`height`をスカラーとして提供します。
  * `class_id`: ボックスのクラスIDを表す整数。`class_labels`キーを参照します。
  * `scores`: スコアの文字列ラベルと数値の辞書。UIでボックスをフィルタリングするのに使用できます。
  * `domain`: ボックス座標の単位/フォーマットを指定します。ボックス座標がピクセル空間（つまり画像の次元内での整数）で表されている場合は**これを「pixel」に設定**します。デフォルトでは、座標が画像の割合/パーセンテージ（0から1までの浮動小数点数）であると仮定します。
  * `box_caption`: （オプション）このボックスのラベルテキストとして表示される文字列
* `class_labels`: （オプション）`class_id`を文字列にマッピングする辞書。デフォルトでは、`class_0`、`class_1`などのクラスラベルを生成します。

以下の例をご覧ください:

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
                    # デフォルトの相対的/小数のドメインで表現された1つのボックス
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # ピクセルドメインで表現された別のボックス
                    # （説明のためのみ、すべてのボックスは同じドメイン/フォーマットである可能性が高い）
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # 必要に応じて多くのボックスをログします
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 意味のあるボックスのグループごとに一意のキー名でログします
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## テーブルでの画像オーバーレイ

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![テーブルでのインタラクティブなセグメンテーションマスク](/images/track/Segmentation_Masks.gif)

テーブルにセグメンテーションマスクをログに記録するには、各テーブル行に`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例が示されています:

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

![テーブルでのインタラクティブなバウンディングボックス](/images/track/Bounding_Boxes.gif)

テーブルにバウンディングボックスを持つ画像をログに記録するには、各テーブル行に`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例が示されています:

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
    {label: 'Basic Histogram Logging', value: 'histogram_logging'},
    {label: 'Flexible Histogram Logging', value: 'flexible_histogram'},
    {label: 'Histograms in Summary', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">

数列（例: リスト、配列、テンソル）が最初の引数として提供されると、自動的に`np.histogram`を呼び出してヒストグラムを構築します。すべての配列/テンソルはフラットにされます。オプションの`num_bins`キーワード引数を使用して、デフォルトの`64`ビンを上書きできます。サポートされるビンの最大数は`512`です。

UIでは、トレーニングステップがx軸、メトリック値がy軸、カウントが色で表されるヒートマップでヒストグラムがプロットされ、トレーニング全体で記録されたヒストグラムの比較が容易になります。「ヒストグラムの概要」タブで詳細を見ることができます。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GANのディスクリミネータの勾配。](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

より細かいコントロールが必要な場合は、`np.histogram`を呼び出し、返されたタプルを`np_histogram`キーワード引数に渡します。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # 概要にのみ記録される場合、概要タブでのみ表示
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

ヒストグラムが概要にある場合、[Runページ](../../app/pages/run-page.md)の概要タブに表示されます。履歴にある場合、チャートタブでビンのヒートマップを時間経過とともにプロットします。

## 3D可視化

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D Object', value: '3d_object'},
    {label: 'Point Clouds', value: 'point_clouds'},
    {label: 'Molecules', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

`'obj'`, `'gltf'`, `'glb'`, `'babylon'`, `'stl'`, `'pts.json'`形式のファイルをログに記録し、runが終了したときにUIでそれらをレンダリングします。

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

[ライブの例を参照 →](https://app.wandb.ai/nbaryd/SparseConvNet-examples_3d_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

3DポイントクラウドやLidarシーンをバウンディングボックスと一緒にログに記録します。表示する点の座標と色を含むNumPy配列を渡します。UIでは、300,000点にトランケートされます。

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

柔軟なカラーリングスキームをサポートする3つの異なるNumPy配列の形状がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c is a category` が `[1, 14]` の範囲にある (セグメンテーションに有用)
* `[[x, y, z, r, g, b], ...]` `nx6 | r, g, b` が `[0,255]` の範囲にある赤、緑、青の色チャネル値

以下の例のログコードを参照してください:

* `points`は上記のシンプルなポイントクラウドレンダラと同じ形式のNumPy配列です
* `boxes`は3つの属性を持つPython辞書のNumPy配列です:
  * `corners` - 8つのコーナーのリスト
  * `label` - ボックスに表示されるラベルを表す文字列（オプション）
  * `color` - ボックスの色を表すRGB値
* `type`はレンダリングするシーンタイプを表す文字列です。現在サポートされている唯一の値は`lidar/beta`です

```python
# W&Bにポイントとボックスをログ
point_scene = wandb.Object3D(
    {
        "type": "lidar/beta",
        "points": np.array(  # 点を追加、ポイントクラウドとして
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

以下の10種類のファイル形式の分子データをログに記録できます: `pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, `mmtf`

W&Bはまた、SMILES文字列、[`rdkit`](https://www.rdkit.org/docs/index.html)の`mol`ファイル、`rdkit.Chem.rdchem.Mol`オブジェクトからの分子データのログ記録もサポートしています。

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

[AlphaFoldを使用したライブの例を見る →](http://wandb.me/alphafold-workspace)

![](@site/static/images/track/docs-molecule.png)
  </TabItem>
</Tabs>

## その他のメディア

W&Bは他の様々なメディアタイプのログ記録もサポートしています。

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

ステップごとにログ記録できる音声クリップの最大数は100です。

  </TabItem>
  <TabItem value="video">

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy配列が供給された場合、次元は順番に時間、チャネル、幅、高さであると仮定します。デフォルトでは、4fpsのgif画像を作成します（numpyオブジェクトを渡す場合、[`ffmpeg`](https://www.ffmpeg.org)と[`moviepy`](https://pypi.org/project/moviepy/)Pythonライブラリが必要です）。サポートされる形式は`"gif"`、`"mp4"`、`"webm"`、`"ogg"`です。文字列を`wandb.Video`に渡す場合、ファイルが存在し、サポートされる形式であることを確認してから、wandbにアップロードします。`BytesIO`オブジェクトを渡すと、指定した形式の拡張子で一時ファイルが作成されます。

W&Bの[Run](../../app/pages/run-page.md)および[Project](../../app/pages/project-page.md)ページで、メディアセクションにビデオが表示されます。

  </TabItem>
  <TabItem value="text">

`wandb.Table`を使用してテーブルでテキストをログに記録し、UIに表示します。デフォルトでは、カラムヘッダーは`["Input", "Output", "Expected"]`です。UIのパフォーマンスを確保するために、デフォルトで最大行数は10,000に設定されていますが、ユーザーは`wandb.Table.MAX_ROWS = {DESIRED_MAX}`を使用して最大行数を明示的に上書きできます。

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

pandasの`DataFrame`オブジェクトを渡すこともできます。

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

カスタムhtmlは任意のキーでログに記録でき、このHTMLパネルがrunページに表示されます。デフォルトでは、スタイルを注入します。`inject=False`を指定することでデフォルトのスタイルを無効にすることができます。

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## よくある質問

### エポックまたはステップにわたって画像やメディアを比較するにはどうすればよいですか？

ステップごとに画像をログに記録するたびに、UIで表示するために保存されます。画像パネルを展開し、ステップスライダーを使用して異なるステップの画像を閲覧します。これにより、モデルの出力がトレーニング中にどのように変化するかを比較しやすくなります。

### プロジェクトにW&Bを統合したいが、画像やメディアをアップロードしたくない場合はどうすればよいですか？

W&Bはスカラーのログ記録のみを行うプロジェクトにも使用できます。アップロードするファイルやデータは明示的に指定します。[画像をログに記録しないPyTorchの例はこちら](http://wandb.me/pytorch-colab)を参照してください。

### PNGをログに記録するにはどうすればよいですか？

[`wandb.Image`](../../../ref/python/data-types/image.md)は、`numpy`配列や`PILImage`インスタンスをデフォルトでPNGに変換します。

```python
wandb.log({"example": wandb.Image(...)})
# または複数の画像
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### ビデオをログに記録するにはどうすればよいですか？

ビデオは[`wandb.Video`](../../../ref/python/data-types/video.md)データタイプを使用してログに記録されます:

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

これでメディアブラウザでビデオを視聴できます。プロジェクトワークスペース、runワークスペース、またはレポートに移動し、「Add visualization」をクリックしてリッチメディアパネルを追加します。

### ポイントクラウドをナビゲートしズームするにはどうすればよいですか？

Ctrlキーを押しながらマウスで空間内を移動します。

### 分子の2Dビューをログに記録するにはどうすればよいですか？

[`wandb.Image`](../../../ref/python/data-types/image.md)データタイプと[`rdkit`](https://www.rdkit.org/docs/index.html)を使用して分子の2Dビューをログに記録できます:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

