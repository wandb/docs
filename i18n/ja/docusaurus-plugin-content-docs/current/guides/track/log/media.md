---
description: 3Dポイントクラウドや分子からHTMLやヒストグラムまで、豊富なメディアをログ化
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# メディアとオブジェクトのログ化

画像、ビデオ、オーディオなど、さまざまなメディアに対応しています。豊富なメディアをログ化して、結果を探索し、run、モデル、データセットを視覚的に比較してみてください。以下に例とハウツーガイドをご紹介します。

:::info
各種メディアタイプのリファレンスドキュメントをお探しですか？[このページ](../../../ref/python/data-types/)をご覧ください。
:::

<!-- {% embed url="https://www.youtube.com/watch?v=96MxRvx15Ts" %} -->

:::info
これらのメディアオブジェクトをすべてログ化するための動作コードは、[このColabノートブック](http://wandb.me/media-colab)でご覧いただけます。wandb.aiでの結果の見た目は[こちら](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)で確認できます。ビデオチュートリアルも上記リンクからご覧いただけます。
:::

## 画像

入力、出力、フィルターの重み、活性化関数などをトラッキングするために画像をログ化しましょう！

![インペインティングを実行するオートエンコーダーネットワークの入力と出力。](/images/track/log_images.png)

画像は、NumPy配列から直接、PIL画像として、またはファイルシステムからログ化できます。

:::info
トレーニング中のログ化がボトルネックにならないように、1ステップあたり50枚以下の画像をログ化することをお勧めします。また、結果を表示する際に画像の読み込みがボトルネックにならないようにするためにもこの枚数にしてください。
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: '配列を画像としてログに記録', value: 'arrays'},
    {label: 'PIL画像をログに記録', value: 'pil_images'},
    {label: 'ファイルからの画像をログに記録', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

手動で画像を作成する際に、例えば [`torchvision` からの `make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid)を使用して、配列を直接提供します。

配列は、[Pillow](https://pillow.readthedocs.io/en/stable/index.html)を使用してpngに変換されます。

```python
images = wandb.Image(
    image_array, 
    caption="Top: Output, Bottom: Input"
    )
          
wandb.log({"examples": images}
```

最後の次元が1の場合、画像をグレースケールと仮定し、3の場合は RGB とし、4の場合は RGBA とします。配列に float が含まれている場合、`0` から `255` までの整数に変換します。画像を異なる方法で正規化したい場合は、[`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) を手動で指定するか、このパネルの "PIL画像をログに記録" タブで説明されているように、[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を提供してください。
  </TabItem>
  <TabItem value="pil_images">

配列を画像に変換する際の完全な制御が必要な場合は、 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) を自分で作成し、直接提供してください。

```python
images = [PIL.Image.fromarray(image) for image in image_array]
こちらのMarkdownテキストを翻訳してください。英語以外の言語は返さず、翻訳されたテキストだけを返してください。テキスト:

wandb.log({"examples": [wandb.Image(image) for image in images]}
```
  </TabItem>
  <TabItem value="images_files">
さらに制御を利かせるために、任意の方法で画像を作成し、ディスクに保存してファイルパスを指定します。

```python
im = PIL.fromarray(...)
rgb_im = im.convert('RGB')
rgb_im.save('myimage.jpg')

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## 画像オーバーレイ

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'セグメンテーションマスク', value: 'segmentation_masks'},
    {label: 'バウンディングボックス', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

W&B UI を介してセマンティックセグメンテーションマスクをログし、それらと対話します（不透明度の変更、時間の経過に伴う変化の表示など）。

![W&B UI でのインタラクティブなマスク表示。](/images/track/semantic_segmentation.gif)
オーバーレイをログに記録するには、`wandb.Image` の `masks` キーワード引数に、次のキーと値を持つ辞書を提供する必要があります。

* 画像マスクを表す2つのキーのうちの1つ：
  * `"mask_data"`：各ピクセルに対して整数のクラスラベルを持つ2D NumPy配列
  * `"path"`：（文字列）保存された画像マスクファイルへのパス
* `"class_labels"`：（オプション）画像マスク内の整数クラスラベルを読みやすいクラス名にマッピングする辞書

複数のマスクをログに記録するには、以下のコードスニペットのように、複数のキーを持つマスク辞書をログに記録します。

[ライブ例を見る →](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[サンプルコード →](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix\_P4J)

```python
mask_data = np.array([[1, 2, 2, ... , 2, 2, 1], ...])

class_labels = {
  1: "tree",
  2: "car",
  3: "road"
}

mask_img = wandb.Image(image, masks={
  "predictions": {
    "mask_data": mask_data,
    "class_labels": class_labels
  },
  "ground_truth": {
    ...
  },
  ...
})
```
  </TabItem>
  <TabItem value="bounding_boxes">
画像とバウンディングボックスをログに記録し、フィルターやトグルを使ってUIで異なるボックスのセットを動的に可視化します。
![](@site/static/images/track/bb-docs.jpeg)

[ライブ例を見る →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

バウンディングボックスをログに記録するには、以下のキーと値を持つ辞書を `wandb.Image` の boxes キーワード引数に提供する必要があります。

* `box_data`：各ボックス用の辞書のリスト。ボックス辞書の形式は以下で説明します。
  * `position`：以下で説明する2つの形式のうちの1つで、ボックスの位置とサイズを表す辞書。すべてのボックスが同じ形式を使用する必要はありません。
    * _オプション1：_ `{"minX", "maxX", "minY", "maxY"}`。各ボックスの次元の上限と下限を定義する座標のセットを提供します。
    * _オプション2：_ `{"middle", "width", "height"}`。 `middle` 座標を `[x, y]` として指定し、 `width` と `height` をスカラーとして指定する座標のセットを提供します。
  * `class_id`：ボックスのクラスIDを表す整数。以下の `class_labels` キーを参照してください。
  * `scores`： スコア用の文字列ラベルと数値値の辞書。UIでボックスをフィルタリングするために使用できます。
  * `domain`: ボックス座標の単位/形式を指定します。** "pixel"に設定してください。**ボックス座標がピクセル空間で表現されている場合（つまり、画像の次元の境界内で整数として）。デフォルトでは、ドメインは画像の分数/パーセンテージ（0から1の浮動小数点数）と仮定されます。
  * `box_caption`:（オプション）このボックスのラベルテキストとして表示される文字列

* `class_labels`:（オプション） `class_id` を文字列にマッピングする辞書。デフォルトでは、 `class_0` 、 `class_1` などのクラスラベルが生成されます。

この例をチェックしてください。

```python
class_id_to_label = {
    1: "car",
    2: "road",
    3: "building",
    ....
}

img = wandb.Image(image, boxes={
    "predictions": {
        "box_data": [{
            # default relative/fractional domainで表現された1つのボックス
            "position": {
                "minX": 0.1,
                "maxX": 0.2,
                "minY": 0.3,
                "maxY": 0.4
            },
            "class_id" : 2,
            "box_caption": class_id_to_label[2],
            "scores" : {
                "acc": 0.1,
                "loss": 1.2
            },
            # ピクセルドメインで表現された別のボックス
            # (説明目的のみで、すべてのボックスが同じドメイン/形式になる可能性があります)
            "position": {
                "middle": [150, 20],
                "width": 68,
                "height": 112
            },
            "domain" : "pixel",
            "class_id" : 3,
            "box_caption": "a building",
            "scores" : {
                "acc": 0.5,
                "loss": 0.7
            },
            ...
            # 必要なだけボックスをログに記録する
        }
        ],
        "class_labels": class_id_to_label
    },
    # 各意味のあるボックスのグループを一意のキー名でログに記録する
    "ground_truth": {
    ...
    }
})
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

テーブル内にセグメンテーションマスクをログするには、各行に対して`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットで例を示しています。

```python
table = wandb.Table(columns=['ID', 'Image'])

for id, img, label in zip(ids, images, labels):
    mask_img = wandb.Image(img, masks = {
        "prediction" : {
            "mask_data" : label,
            "class_labels" : class_labels
        },
        ...
    })
table.add_data(id, img)

wandb.log({"Table" : table})
```
  </TabItem>
  <TabItem value="bounding_boxes">


![テーブル内のインタラクティブなバウンディングボックス](/images/track/Bounding_Boxes.gif)

テーブルにバウンディングボックス付きの画像をログするには、テーブルの各行に`wandb.Image`オブジェクトを提供する必要があります。

以下のコードスニペットに例が示されています。

```python
table = wandb.Table(columns=['ID', 'Image'])

for id, img, boxes in zip(ids, images, boxes_set):
    box_img = wandb.Image(img, boxes = {
        "prediction" : {
            "box_data" : [{
                "position" :{
                    "minX" : box["minX"],
                    "minY" : box["minY"],
                    "maxX" : box["maxX"],
                    "maxY" : box["maxY"]
                },
                "class_id" : box["class_id"],
                "box_caption" : box["caption"],
                "domain" : "pixel"
            } 
            for box in boxes
        ],
        "class_labels" : class_labels
        }
    })
```
  </TabItem>
</Tabs>
## ヒストグラム

<Tabs
  defaultValue="histogram_logging"
  values={[
    {label: '基本的なヒストグラムのログ記録', value: 'histogram_logging'},
    {label: '柔軟なヒストグラムのログ記録', value: 'flexible_histogram'},
    {label: 'サマリー内のヒストグラム', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">

数値のシーケンス（例: リスト、配列、テンソル）が最初の引数として提供される場合、`np.histogram`を呼び出すことで自動的にヒストグラムが作成されます。注意していただきたいのは、すべての配列/テンソルが平坦化されることです。デフォルトの`64`ビンを上書きするために、オプションの`num_bins`キーワード引数を使用することができます。サポートされているビンの最大数は`512`です。

UIでは、ヒストグラムはトレーニングステップをx軸に、メトリック値をy軸に、カウントを色で表現し、トレーニング全体でログ記録されたヒストグラムの比較が容易になるようにプロットされます。一度だけのヒストグラムをログに記録する詳細については、このパネルの"Histograms in Summary"タブを参照してください。

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GANのディスクリミネータの勾配](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

もし、より多くのコントロールが必要な場合、`np.histogram`を呼び出し、返されたタプルを`np_histogram`キーワード引数に渡してください。

```python
np_hist_grads = np.histogram(grads, density=True, range=(0., 1.))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # サマリーのみの場合、概要タブでのみ表示される
  {"final_logits": wandb.Histogram(logits)})
```

  </TabItem>
</Tabs>

ヒストグラムがサマリーに含まれている場合、[Runページ](../../app/pages/run-page.md) の概要タブに表示されます。ヒストリに含まれている場合は、チャートタブに時間をかけたビンのヒートマップが表示されます。

## 3D可視化

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3Dオブジェクト', value: '3d_object'},
    {label: 'ポイントクラウド', value: 'point_clouds'},
    {label: '分子', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

ログファイルは、`'obj'`, `'gltf'`, `'glb'`, `'babylon'`, `'stl'`, `'pts.json'`の形式で、ランが終了したときにUIでレンダリングされます。

```python
wandb.log({"generated_samples":
           [wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb"))]})
```

![ヘッドフォンのポイントクラウドの正解と予測](/images/track/ground_truth_prediction_of_3d_point_clouds.png)
[ライブ例を見る →](https://app.wandb.ai/nbaryd/SparseConvNet-examples\_3d\_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

3DポイントクラウドやLidarシーンにバウンディングボックスを付けてログを取ります。レンダリングするポイントの座標と色を含むNumPy配列を渡します。UIでは、30万ポイントに切り捨てます。

```python
point_cloud = np.array([[0, 0, 0, COLOR...], ...])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

柔軟なカラースキームに対応するために、3つの異なる形状のNumPy配列がサポートされています。

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| cはカテゴリ` の範囲は `[1, 14]` (セグメンテーションに便利)
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` は赤、緑、青のカラーチャンネルの範囲 `[0,255]` の値です。

以下はログ記録コードの例です:

* `points` は、上記のシンプルなポイントクラウドレンダラと同じフォーマットのNumPy配列です。
* `boxes` は、3つの属性を持つPython辞書のNumPy配列です:
  * `corners` - 8つの角のリスト
  * `label` - ボックス上にレンダリングされるラベルを表す文字列 (オプション)
  * `color` - ボックスの色を表すrgb値
* `type` は、レンダリングするシーンタイプを表す文字列です。現在サポートされている唯一の値は `lidar/beta` です

```python
# W&Bにポイントとボックスをログ
point_scene = wandb.Object3D({
    "type": "lidar/beta",
    "points": np.array(  # ポイントクラウドのようにポイントを追加
        [
            [0.4, 1, 1.3], 
            [1, 1, 1], 
            [1.2, 1, 1.2]
        ]
    ),
    "boxes": np.array(  # 3Dボックスを描画
        [
            {
                "corners": [
                    [0,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,0,0],
                    [1,1,0],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1]
                ],
                "label": "Box",
                "color": [123, 321, 111],
            },
            {
                "corners": [
                    [0,0,0],
                    [0,2,0],
                    [0,0,2],
                    [2,0,0],
                    [2,2,0],
                    [0,2,2],
                    [2,0,2],
                    [2,2,2]
                ],
                "label": "Box-2",
                "color": [111, 321, 0],
            }
        ]
      ),
      "vectors": np.array(  # 3Dベクターを追加
          [
              {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5]}
          ]
      )
})
wandb.log({"point_scene": point_scene})
```
  </TabItem>
  <TabItem value="molecules">


```python
wandb.log({"protein": wandb.Molecule("6lu7.pdb")}
```


10種類のファイルタイプの分子データをログすることができます:`pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, `mmtf`.

Weights & Biasesは、SMILES文字列、[`rdkit`](https://www.rdkit.org/docs/index.html) `mol`ファイル、および`rdkit.Chem.rdchem.Mol`オブジェクトからの分子データのログもサポートしています。

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

runが終了すると、UIで分子の3D可視化と対話することができます。

[AlphaFoldを使用したライブ例を見る →](http://wandb.me/alphafold-workspace)

![](@site/static/images/track/docs-molecule.png)
  </TabItem>
</Tabs>

## その他のメディア
Weights & Biasesは、さまざまなメディアタイプのログもサポートしています。

<Tabs
  defaultValue="audio"
  values={[
    {label: 'オーディオ', value: 'audio'},
    {label: 'ビデオ', value: 'video'},
    {label: 'テキスト', value: 'text'},
    {label: 'HTML', value: 'html'},
  ]}>
  <TabItem value="audio">

```python
wandb.log({
    "クジラの歌": wandb.Audio(
        np_array, 
        caption="OooOoo", 
        sample_rate=32)})  
```

ステップごとにログできるオーディオクリップの最大数は100です。

  </TabItem>
  <TabItem value="video">

```python
wandb.log(
  {"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```
もしnumpy配列が与えられた場合、次の順番で次元が与えられると仮定します：時間、チャンネル、幅、高さ。デフォルトでは、4 fpsのgif画像を作成します（numpyオブジェクトを渡す場合、[`ffmpeg`](https://www.ffmpeg.org) と [`moviepy`](https://pypi.org/project/moviepy/) Pythonライブラリが必要です）。対応しているフォーマットは、`"gif"`、`"mp4"`、`"webm"`、`"ogg"`です。`wandb.Video` に文字列を渡すと、ファイルが存在し、対応するフォーマットであることを確認してからwandbにアップロードされます。`BytesIO`オブジェクトを渡すと、指定されたフォーマットを拡張子として一時ファイルが作成されます。

W&Bの[Run](../../app/pages/run-page.md)ページと[Project](../../app/pages/project-page.md)ページでは、メディアセクションにあなたの動画が表示されます。

  </TabItem>
  <TabItem value="text">

UIに表示されるテーブル内のテキストをログに記録するには、`wandb.Table`を使用します。デフォルトでは、列のヘッダーは `["Input", "Output", "Expected"]` です。最適なUIパフォーマンスを確保するために、デフォルトの最大行数は10,000に設定されています。ただし、ユーザーは `wandb.Table.MAX_ROWS = {DESIRED_MAX}` を明示的に上書きして最大値を変更することができます。

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# 方法 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table =  wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# 方法 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

また、pandas の `DataFrame` オブジェクトを渡すこともできます。

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

カスタムHTMLは任意のキーに記録することができ、これによってrunページにHTMLパネルが表示されます。デフォルトではデフォルトのスタイルが挿入されますが、`inject=False`を渡すことでデフォルトのスタイルを無効にすることができます。

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## よくある質問

### エポックやステップをまたいで画像やメディアを比較する方法は？

ステップから画像をログするたびに、それらをUIで表示するために保存します。画像パネルを展開し、ステップスライダーを使用して、異なるステップの画像を表示します。これにより、トレーニング中にモデルの出力がどのように変化するかを簡単に比較することができます。

### プロジェクトにW&Bを統合したいが、画像やメディアをアップロードしたくない場合は？

W&Bは、スカラーのみをログするプロジェクトでも使用できます。明示的にアップロードするファイルやデータを指定します。画像をログしない[PyTorchの簡単な例](http://wandb.me/pytorch-colab)がこちらになります。

### PNGを記録する方法は？

[`wandb.Image`](../../../ref/python/data-types/image.md)は、デフォルトで`numpy`配列や`PILImage`のインスタンスをPNGに変換します。

```python
wandb.log({"example": wandb.Image(...)})
# 複数の画像を表示

wandb.log({"example": [wandb.Image(...) for img in images]})

```



### 動画をログにする方法は？

動画は、[`wandb.Video`](../../../ref/python/data-types/video.md)データ型を使ってログに記録されます:

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```



これでメディアブラウザで動画を表示できます。プロジェクトのワークスペース、runのワークスペース、またはレポートに移動し、「可視化を追加」をクリックしてリッチメディアパネルを追加してください。



### 点群をナビゲートしてズームする方法は？

コントロールキーを押しながらマウスを使って空間内を移動できます。

### 分子の2Dビューをログにする方法は？
[`wandb.Image`](../../../ref/python/data-types/image.md)データ型と[`rdkit`](https://www.rdkit.org/docs/index.html)を使用して、分子の2Dビューをログに記録できます:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```
